"""Create several models and make them vote."""

# Imports of the project
import showing_infos_Windstorm
import import_set
import LSTMwithKeras
from Extractions_carac import diff_between_frames
from Extractions_carac import colours
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


def create_assemble_model(learning_rate, input_size, n_classes, decay):
    """Create model to fit on predictions done by other models.

    The voting model choosen here is a MLP.
    Inputs : learning_rate(float), the learning_rate assigned to the optimizer
             input_size(int), the size of the input to train on
             n_classes(int), the final number of classes to classify on
             decay(int), I don't know what this does
    """
    model = Sequential()
    model.add(Dense(50, activation='softmax', input_shape=(input_size,)))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(n_classes, activation='sigmoid'))

    optimizer = Adam(lr=learning_rate, decay=decay)  #
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    try:
        model.load_weights('../Models/assemble.hdf5')
    except:
        print("Weights of the MLP assemble model have not been found.")
    return model


def fitting_colour_model(n_classes, X_train_colours, X_test_colours, Y_train,
                         Y_test):
    """Create and fit model to be trained on colours.

    Currently, does not give good results at all.
    """
    checkpointer_colours = ModelCheckpoint(
        filepath='../Models/colours.hdf5', verbose=1,
        save_best_only=True)
    CNN_colours = colours.create_model(
        learning_rate=5 * 10 ** -6, n_classes=n_classes)
    print("fitting on colour model")
    array_test_colours = np.array(X_test_colours)
    history = CNN_colours.fit(np.array(X_train_colours), Y_train,
                              batch_size=400,
                              validation_data=(array_test_colours, Y_test),
                              epochs=1000, verbose=1, shuffle=True,
                              callbacks=[checkpointer_colours,
                                         EarlyStopping(patience=100)])
    showing_infos_Windstorm.show_accuracy_over_time(history, "colour model")
    Y_test_pred_CNN_colours = CNN_colours.predict(np.array(X_test_colours))
    print(accuracies_metrics.mat_conf(Y_test, Y_test_pred_CNN_colours))
    CNN_colours.load_weights("../Models/colours.hdf5")
    Y_train_pred_CNN_colours = CNN_colours.predict(np.array(X_train_colours))
    return Y_train_pred_CNN_colours


def fitting_pixel_difference_model(n_classes, X_train_pixelDifference, Y_train,
                                   X_test_pixelDifference, Y_test):
    """Create and fit model to be trained on pixel difference model.

    The feature extraction is done by putting all frames into black and white,
    and by only writing the colour difference between two frames.
    Currently, does not give good results at all.
    """
    checkpointer_pixel_diff = ModelCheckpoint(
        filepath='../Models/pixel_diff.hdf5',
        verbose=1, save_best_only=True)
    CNN = diff_between_frames.create_model(
        learning_rate=10 ** -7, n_classes=n_classes)
    print("fitting on pixel difference model")
    X_train = np.array(X_train_pixelDifference)
    validation_data = (np.array(X_test_pixelDifference), np.array(Y_test))
    history = CNN.fit(X_train, Y_train, validation_data=validation_data,
                      batch_size=400, epochs=1000, verbose=1, shuffle=True,
                      callbacks=[checkpointer_pixel_diff,
                                 EarlyStopping(patience=100)])
    showing_infos_Windstorm.show_accuracy_over_time(
        history, "pixel difference model")
    Y_test_pred_CNN_pixel_diff = CNN.predict(
        np.array(X_test_pixelDifference))
    print(accuracies_metrics.mat_conf(Y_test, Y_test_pred_CNN_pixel_diff))

    CNN.load_weights("../Models/pixel_diff.hdf5")
    Y_train_pred_CNN_pixel_diff = CNN.predict(X_train)
    return Y_train_pred_CNN_pixel_diff


def get_checkpointer(name):
    """Return the checkpointer corresponding to the model."""
    return ModelCheckpoint(filepath=get_path(
        "Models", str(name) + '.hdf5'), verbose=1, save_best_only=True)


def reduce_features(X, nb_final_features, start):
    """Select only a bunch of features from the X.

    Used to choose on which features to train on.
    """
    a = int(start)
    b = int(start + nb_final_features)
    print(a, b)

    new_X = []
    for grp_frames in X:
        grp_frames = np.array(grp_frames)

        new_X.append(grp_frames[:, a:b])
    return new_X


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
    LSTM_CNN = LSTMwithKeras.lstm(
        shape=(15, nb_features), n_classes=n_classes,
        learning_rate=10 ** -4, decay=0.01)
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

        X_train_CNN_local = reduce_features(
            X_train_CNN, nb_features, start=start)
        X_test_CNN_local = reduce_features(
            X_test_CNN, nb_features, start=start)

        LSTM_CNN.optimizer.lr.assign(10 ** -4)
        if i in models_to_fit:
            # X_test_CNN_local, Y_test = shuffle(X_test_CNN_local, Y_test)
            LSTM_CNN.fit(np.array(X_train_CNN_local), Y_train,
                         validation_data=(np.array(X_test_CNN_local), Y_test),
                         batch_size=500, epochs=300, verbose=1, shuffle=True,
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
    return train_preds, test_preds, accs


# ################Creating models ###########################


def append_to_file(acc_vote_prob, acc_vote_class, acc_MLP,
                   acc_boosting, acc_MLP_class, acc_boosting_MLP_class, nb_features, nb_models):
    """Write the result of one exp into a text file."""
    with open('info_files/accuracies.txt', 'a') as f:
        to_write = "acc_vote_prob: {0} acc_vote_class: {1} acc_MLP: {2} acc_boosting: {3} acc_MLP_class: {4} acc_boosting_MLP_class: {5} nb_models: {6} nb_features: {7} \n".format(
            str(acc_vote_prob), str(acc_vote_class), str(acc_MLP),
            str(acc_boosting), str(acc_MLP_class), str(nb_models),
            str(nb_features))
        f.write(to_write)
        f.close()


def get_accs(Y_test, Y_test_pred_assemble, Y_test_pred_vote,
             Y_test_pred_average_proba, n_models, Y_test_pred_boosting_MLP,
             Y_test_pred_MLP_on_classes, Y_test_pred_boosting_MLP_on_classes):
    """Get accs of different voting methods."""
    acc_vote_prob = accuracies_metrics.mat_conf(
        Y_test, Y_test_pred_assemble, "vote with probas predicted")
    acc_vote_class = accuracies_metrics.mat_conf(
        Y_test, Y_test_pred_vote, "vote with class predicted")
    acc_MLP = accuracies_metrics.mat_conf(
        Y_test, Y_test_pred_average_proba, " MLP on probas predicted")
    acc_boosting = 0
    acc_boosting_MLP_class = 0
    if n_models != 1:
        acc_boosting = accuracies_metrics.mat_conf(
            Y_test, Y_test_pred_boosting_MLP, "boosting on probas predicted")
        acc_boosting_MLP_class = accuracies_metrics.mat_conf(
            Y_test, Y_test_pred_boosting_MLP_on_classes, "boosting on probas predicted")
    acc_MLP_class = accuracies_metrics.mat_conf(
        Y_test, Y_test_pred_MLP_on_classes, "MLP on class, predicted")

    return acc_vote_prob, acc_vote_class, acc_MLP, acc_boosting, acc_MLP_class, acc_boosting_MLP_class


def one_iter(n_classes, n_models, nb_features, X_train_CNN, X_test_CNN,
             X_train_pixelDifference, X_test_pixelDifference, X_train_colours,
             X_test_colours, Y_train, Y_test, groups_frames_names_train,
             groups_frames_names_test):
    """Test all voting methods once LSTM model have been trained."""
    showing_infos_Windstorm.infos(Y_test, Y_train)
    Y_train, Y_test = np.array(Y_train), np.array(Y_test)

    checkpointer_assemble = ModelCheckpoint(
        filepath='../Models/assemble.hdf5', verbose=1,
        save_best_only=True, monitor='val_acc')

    checkpointer_MLP_class_predicted = get_checkpointer(
        "MLP_on_classes_predicted")
    checkpointer_assemble = get_checkpointer("assemble")
    models_to_fit = [i for i in range(n_models)]
    models_to_fit = []
    train_preds, test_preds, accs = bagging_LSTM(
        n_models, n_classes, models_to_fit, X_train_CNN, X_test_CNN,
        Y_train, Y_test, nb_features)

    X_train_assemble = import_set.get_assemble_set(
        train_preds, n_classes, Y_train)
    X_test_assemble = import_set.get_assemble_set(
        test_preds, n_classes, Y_train)

    X_train_classes_predicted = voting_methods.X_probas_to_X_class(
        X_train_assemble, n_models, n_classes)
    X_test_classes_predicted = voting_methods.X_probas_to_X_class(
        X_test_assemble, n_models, n_classes)
    # ###################### Assemble Learning ##############################

    print("assemble model")
    model = create_assemble_model(learning_rate=10 ** -3,
                                  input_size=n_models * n_classes,
                                  n_classes=n_classes, decay=0.01)
    validation_data = (X_test_assemble, np.array(Y_test))
    callbacks = [checkpointer_assemble, EarlyStopping(patience=300)]
    history = model.fit(X_train_assemble, Y_train,
                        validation_data=validation_data, batch_size=300,
                        epochs=1800, verbose=1, shuffle=True,
                        callbacks=callbacks)
    model.load_weights('../Models/assemble.hdf5')
    Y_test_pred_assemble = model.predict(X_test_assemble)

    print("MLP on class predicted model")
    MLP_on_classes_predicted = create_assemble_model(
        learning_rate=10 ** -3, input_size=n_models,
        n_classes=n_classes, decay=0.01)
    validation_data = (X_test_classes_predicted, np.array(Y_test))
    callbacks = [checkpointer_MLP_class_predicted, EarlyStopping(patience=300)]
    history = MLP_on_classes_predicted.fit(X_train_classes_predicted, Y_train, validation_data=validation_data,
                                           batch_size=300, epochs=1800, verbose=1,
                                           shuffle=True, callbacks=callbacks)
    # showing_infos_Windstorm.show_accuracy_over_time(
    #   history, "assemble model")
    MLP_on_classes_predicted.load_weights(
        '../Models/MLP_on_classes_predicted.hdf5')

    Y_test_pred_vote = voting_methods.more_vote_wins(X_test_assemble, n_models, n_classes, accs)
    Y_test_pred_average_proba = voting_methods.more_prob_wins(X_test_assemble, n_models, n_classes, accs)
    Y_test_pred_MLP_on_classes = MLP_on_classes_predicted.predict(X_test_classes_predicted)
    if n_models != 1:
        Y_test_pred_boosting_MLP = voting_methods.boosting_MLP(
            X_train_assemble, X_test_assemble, Y_train, Y_test, n_models, n_classes)

        Y_test_pred_boosting_MLP_on_classes = voting_methods.boosting_MLP_class_predicted(
            np.array(X_train_assemble), X_test_assemble, Y_train, Y_test, n_models, n_classes, n_models_together=2)

    del model, history
    gc.collect()

    acc_vote_prob, acc_vote_class, acc_MLP, acc_boosting, acc_MLP_class, acc_boosting_MLP_class = get_accs(
        Y_test, Y_test_pred_assemble, Y_test_pred_vote,
        Y_test_pred_average_proba, n_models, Y_test_pred_boosting_MLP,
        Y_test_pred_MLP_on_classes, Y_test_pred_boosting_MLP_on_classes)
    return acc_vote_prob, acc_vote_class, acc_MLP, acc_boosting, acc_MLP_class, acc_boosting_MLP_class, nb_features, n_models


def main():
    """Launch the global programm."""
    names_train = import_set.get_names_train()
    n_classes = 3
    n_models = [2]
    nb_features = [512]
    # data_processing.get_all_names_from_path("Windstorm/videos")
    all_names = names_train
    names_wanted = all_names

    X_train_CNN, X_test_CNN, X_train_pixelDifference, X_test_pixelDifference, X_train_colours, X_test_colours, Y_train, Y_test, groups_frames_names_train, groups_frames_names_test = import_set.make_train_test_set(
        *import_set.get_set(load_pickle=True, names_wanted=names_wanted),
        percentage_testset=0.2, names_train_imposed=names_train)

    for n_model in n_models:
        for nb_feature in nb_features:
            acc_vote_prob, acc_vote_class, acc_MLP, acc_boosting, acc_MLP_class, acc_boosting_MLP_class, nb_features, n_models = one_iter(
                n_classes, n_model, nb_feature, X_train_CNN, X_test_CNN,
                X_train_pixelDifference, X_test_pixelDifference,
                X_train_colours, X_test_colours, Y_train, Y_test,
                groups_frames_names_train, groups_frames_names_test)
            append_to_file(acc_vote_prob, acc_vote_class, acc_MLP,
                           acc_boosting, acc_MLP_class, acc_boosting_MLP_class, nb_features, n_models)


if __name__ == '__main__':
    main()
