"""Create several models and make them vote."""

# Imports of the project
import showing_infos_Windstorm
import import_set
import LSTMwithKeras

import voting_methods
import accuracies_metrics
from data_processing import get_path

# General imports
import numpy as np
import gc
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import Sequential
from keras.layers import Dense

from keras.optimizers import Adam


def create_MLP_model(lr, input_size, n_classes, decay):
    """Create model to fit on predictions done by other models.

    The voting model choosen here is a MLP.
    Inputs : lr(float), the learning_rate assigned to the optimizer
             input_size(int), the size of the input to train on
             n_classes(int), the final number of classes to classify on
             decay(int), I don't know what this does
    """
    model = Sequential()
    model.add(Dense(50, activation='softmax', input_shape=(input_size,)))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(n_classes, activation='sigmoid'))

    optimizer = Adam(lr=lr, decay=decay)  #
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def predict_with_MLP(input_size, nb_class, assemble_set, Y_train, Y_test, message):
    """Create a MLP and fit on the set given.

    Return the prediction made with assemble set.
    """
    checkpointer = get_checkpointer("MLP")
    showing_infos_Windstorm.print_with_lines(message)
    MLP_on_proba_predicted = create_MLP_model(lr=2 * 10 ** -3, input_size=input_size, n_classes=nb_class,
                                              decay=0.01)
    validation_data = (assemble_set.X_test, np.array(Y_test))
    callbacks = [checkpointer, EarlyStopping(patience=300)]
    history = MLP_on_proba_predicted.fit(assemble_set.X_train, Y_train, validation_data=validation_data, batch_size=300,
                                         epochs=1800, verbose=0, shuffle=True, callbacks=callbacks)
    try:
        MLP_on_proba_predicted.load_weights('../Models/assemble.hdf5')
    except:
        print("Weights have not been found.")
    Y_pred = MLP_on_proba_predicted.predict(assemble_set.X_test)
    return Y_pred


def get_checkpointer(name):
    """Return the checkpointer corresponding to the model."""
    return ModelCheckpoint(filepath=get_path("Models", str(name) + '.hdf5'), verbose=1, save_best_only=True)


def reduce_features(X, nb_final_features, start):
    """Select only a bunch of features from the X.

    Used to choose on which features to train on.
    """
    a, b = int(start), int(start + nb_final_features)
    return [np.array(grp_frames)[:, a:b] for grp_frames in X]


def load_weights_LSTM_CNN(model, name_file):
    """Try to load trained weights, else, loads initial weights."""
    try:
        model.load_weights(get_path("Models", str(name_file) + '.hdf5'))
    except:
        print("Weights have not been found.")
        model.load_weights(get_path("Models", "Initial_weights_LSTM_CNN.hdf5"))


def bagging_LSTM(n_models, n_classes, models_to_fit, X_train_CNN, X_test_CNN,
                 Y_train, Y_test, nb_features):
    """Train each LSTM model on their respective train set."""
    print("fitting on LSTM model")
    early_stopper = EarlyStopping(patience=7)

    train_preds = []
    test_preds = []
    accs = []
    LSTM_CNN = LSTMwithKeras.lstm(shape=(15, nb_features), n_classes=n_classes, learning_rate=10 ** -4, decay=0.01)
    path_initial_weights = get_path("Models", "Initial_weights_LSTM_CNN.hdf5")
    LSTM_CNN.save_weights(path_initial_weights)
    for i in range(n_models):
        print("model number : {0}".format(str(i + 1)))

        name_file = "model_{0}".format(str(i))

        load_weights_LSTM_CNN(LSTM_CNN, name_file)
        checkpointer = get_checkpointer(name_file)

        n_models_fixed = 8
        if n_models != 1:
            start = (i / (n_models_fixed - 1)) * (2048 - nb_features)
        else:
            start = 0

        X_train_CNN_local = reduce_features(X_train_CNN, nb_features, start=start)
        X_test_CNN_local = reduce_features(X_test_CNN, nb_features, start=start)

        LSTM_CNN.optimizer.lr.assign(10 ** -4)
        if i in models_to_fit:
            # X_test_CNN_local, Y_test = shuffle(X_test_CNN_local, Y_test)
            LSTM_CNN.fit(np.array(X_train_CNN_local), Y_train,
                         validation_data=(np.array(X_test_CNN_local), Y_test),
                         batch_size=500, epochs=300, verbose=0, shuffle=True,
                         callbacks=[checkpointer, early_stopper])
            load_weights_LSTM_CNN(LSTM_CNN, name_file)
        Y_train_pred_LSTM, Y_test_pred_LSTM = LSTM_CNN.predict(
            np.array(X_train_CNN_local)), LSTM_CNN.predict(
            np.array(X_test_CNN_local))
        acc = accuracies_metrics.mat_conf(Y_test, Y_test_pred_LSTM)

        accs.append(acc)
        train_preds.append(Y_train_pred_LSTM)
        test_preds.append(Y_test_pred_LSTM)
    del LSTM_CNN, checkpointer

    return import_set.Data_sets(train_preds, test_preds), accs


# ################Creating models ###########################


class Parameters:
    def __init__(self, nb_model, nb_class, names_train, nb_feature, models_to_fit):
        self.nb_model = nb_model
        self.nb_class = nb_class
        self.names_train = names_train
        self.nb_feature = nb_feature
        self.models_to_fit = models_to_fit


class Accuracies:
    def __init__(self, acc_vote_prob, acc_vote_class, acc_MLP, acc_boosting, acc_MLP_class, acc_boosting_MLP_class):
        self.legend_acc = {}
        self.legend_acc[" acc_vote_prob : "] = acc_vote_prob
        self.legend_acc[" acc_vote_class : "] = acc_vote_class
        self.legend_acc[" acc_MLP : "] = acc_MLP
        self.legend_acc[" acc_boosting : "] = acc_boosting
        self.legend_acc[" acc_MLP_class : "] = acc_MLP_class
        self.legend_acc[" acc_boosting_MLP_class : "] = acc_boosting_MLP_class

    def append_to_file(self, parameters):
        self.legend_acc[" nb_features : "] = parameters.nb_feature
        self.legend_acc[" nb_models : "] = parameters.nb_model
        with open('info_files/accuracies.txt', 'a') as f:
            f.write(''.join(legend + str(acc_value) for legend, acc_value in self.legend_acc.items()) + '\n')


def one_iter(cnn_features_extracted, Y_train, Y_test, parameters):
    """Test all voting methods once LSTM model have been trained."""
    nb_class = parameters.nb_class
    nb_model = parameters.nb_model
    nb_feature = parameters.nb_feature

    try:
        showing_infos_Windstorm.infos(Y_test, Y_train)
    except:
        print("you need to have at least one video of each label.")
    Y_train, Y_test = np.array(Y_train), np.array(Y_test)

    predictions, accs = bagging_LSTM(
        nb_model, nb_class, parameters.models_to_fit, cnn_features_extracted.X_train, cnn_features_extracted.X_test,
        Y_train, Y_test, nb_feature)

    assemble_set = import_set.get_assemble_set(predictions)
    assemble_class_set = import_set.Data_sets(
        voting_methods.X_probas_to_X_class(assemble_set.X_train, nb_model, nb_class),
        voting_methods.X_probas_to_X_class(assemble_set.X_test, nb_model, nb_class)
    )

    # ###################### Assemble Learning ##############################

    Y_test_pred_proba = predict_with_MLP(
        nb_model * nb_class, nb_class, assemble_set, Y_train, Y_test,
        message="MLP model : a MLP learning on previous predictions (probabilities) ")
    Y_test_pred_MLP_on_classes = predict_with_MLP(
        nb_model, nb_class, assemble_class_set, Y_train, Y_test,
        message="MLP model : MLP learning on class predicted ")

    Y_test_pred_vote = voting_methods.more_vote_wins(assemble_set.X_test, nb_model, nb_class, accs)
    Y_test_pred_average_proba = voting_methods.more_prob_wins(assemble_set.X_test, nb_model, nb_class, accs)
    if nb_model != 1:
        showing_infos_Windstorm.print_with_lines("Boosting with MLP : boosting with probabilities predicted")
        Y_test_pred_boosting_MLP = voting_methods.boosting_MLP(
            assemble_set.X_train, assemble_set.X_test, Y_train, Y_test, nb_model, nb_class)

        showing_infos_Windstorm.print_with_lines("Boosting with MLP : boosting with class predicted")
        Y_test_pred_boosting_MLP_on_classes = voting_methods.boosting_MLP_class_predicted(
            np.array(assemble_set.X_train), assemble_set.X_test, Y_train, Y_test, nb_model, nb_class,
            n_models_together=2)

    accuracies = Accuracies(
        *showing_infos_Windstorm.get_accs(
            Y_test, Y_test_pred_proba, Y_test_pred_vote, Y_test_pred_average_proba, nb_model, Y_test_pred_boosting_MLP,
            Y_test_pred_MLP_on_classes, Y_test_pred_boosting_MLP_on_classes))
    return accuracies


def main():
    """Launch the global programm."""

    nb_models = [3]
    nb_features = [512]
    for nb_model in nb_models:
        for nb_feature in nb_features:
            parameters = Parameters(nb_model=nb_model, nb_class=3, names_train=import_set.get_names_train(),
                                    nb_feature=nb_feature, models_to_fit=[])

            cnn_features_extracted, pixel_difference, colours, Y_train, Y_test, groups_frames_names_train, groups_frames_names_test = import_set.make_train_test_set(
                *import_set.get_set(load_pickle=False, parameters=parameters), percentage_testset=0.2, parameters=parameters)

            accuracies = one_iter(cnn_features_extracted, Y_train, Y_test, parameters)
            accuracies.append_to_file(parameters)


if __name__ == '__main__':
    main()
