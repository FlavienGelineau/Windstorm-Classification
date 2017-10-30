"""Vote methods are defined here."""
from data_processing import get_path

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def more_vote_wins(X, n_models, n_classes, accuracies):
    """Votes averaged by accuracies of each model."""
    res = []
    for line in X:
        preds = []
        for i in range(n_models):
            sub_pred = [line[3 * i], line[3 * i + 1], line[3 * i + 2]]
            pred = sub_pred.index(max(sub_pred))
            preds.append(pred)
        pred_classes = [0, 0, 0]
        for i in range(len(preds)):
            pred_classes[preds[i]] += accuracies[i]
        res.append(pred_classes.index(max(pred_classes)))

    return res


def more_prob_wins(X, n_models, n_classes, accuracies):
    """Probas predicted averaged by accuracies of each model."""
    res = []
    for line in X:
        preds = [0, 0, 0]
        for i in range(n_models):
            for j in range(n_classes):
                preds[j] += line[i * n_classes + j] * accuracies[i]
        res.append(preds)

        final_predictions = [elt.index(max(elt)) for elt in res]
    return final_predictions


def create_vote_model(learning_rate, input_size, n_classes, decay):
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
        model.load_weights(get_path("Models", "assemble.hdf5"))
    except:
        print("Weights of the MLP assemble model have not been found.")
    return model


def X_probas_to_X_class(X, n_models, n_classes):
    """Create an assemble model which learn on votes (and not probas).

    Currently, the voting system associated with this function does't work.
    """
    res = []
    for line in X:
        preds = []
        for i in range(n_models):
            sub_pred = [line[3 * i], line[3 * i + 1], line[3 * i + 2]]
            pred = sub_pred.index(max(sub_pred))
            preds.append(pred)
        assert len(preds) == n_models

        res.append(preds)

    return res


def concat(l1, l2):
    return np.array([l1[i] + l2[i] for i in range(len(l1))])


def boosting_MLP(X_train, X_test, Y_train, Y_test, n_models, n_classes):
    """Votes taken as an input into a MLP."""

    def get_X_boosting(a, b, X, n_classes):
        return [X[i][a:b] for i in range(len(X))]

    X_train_boosting = get_X_boosting(0, 2 * n_classes, X_train, n_classes)
    X_test_boosting = get_X_boosting(0, 2 * n_classes, X_test, n_classes)

    for i in range(2, n_models + 1):

        input_size = 2 * n_classes

        model = create_and_fit_model_boost(
            input_size, n_classes, X_test_boosting, Y_test,
            X_train_boosting, Y_train)
        Y_train_pred = model.predict(X_train_boosting).tolist()
        Y_test_pred = model.predict_classes(X_test_boosting)
        if i != n_models:
            new_X_train_boosting = get_X_boosting(i * n_classes, (i + 1) * n_classes, X_train, n_classes)
            new_X_test_boosting = get_X_boosting(i * n_classes, (i + 1) * n_classes, X_test, n_classes)

            X_train_boosting = concat(Y_train_pred, new_X_train_boosting)
            X_test_boosting = concat(Y_test_pred, new_X_test_boosting)

    return Y_test_pred


def get_X_boosting_classes(X, n_classes, n_models_together):
    return [[
        np.array(X[i][j * n_classes:(j + 1) * n_classes]).argmax()
        for j in range(n_models_together)]
        for i in range(len(X))]


def create_and_fit_model_boost(
        input_size, n_classes, X_test_boosting, Y_test, X_train_boosting,
        Y_train):
    model = create_vote_model(
        learning_rate=10 ** -3, input_size=input_size,
        n_classes=n_classes, decay=0)
    checkpointer_boosting = ModelCheckpoint(
        filepath=get_path("Models", "boosting_vote.hdf5"), verbose=1,
        save_best_only=True, monitor='val_acc')

    validation_data = (X_test_boosting, np.array(Y_test))
    callbacks = [checkpointer_boosting, EarlyStopping(patience=800)]
    model.fit(X_train_boosting, Y_train, validation_data=validation_data,
              batch_size=800, epochs=1800, verbose=1, shuffle=True,
              callbacks=callbacks)
    model.load_weights('../Models/boosting_vote.hdf5')
    return model


def boosting_MLP_class_predicted(X_train, X_test, Y_train, Y_test,
                                 n_models, n_classes, n_models_together=2):
    """Votes taken as an input into a MLP."""
    X_train_boosting = get_X_boosting_classes(X_train, n_classes, n_models_together)
    X_test_boosting = get_X_boosting_classes(X_test, n_classes, n_models_together)

    for i in range(2, n_models + 1):

        model = create_and_fit_model_boost(n_models_together, n_classes, X_test_boosting, Y_test, X_train_boosting,
                                           Y_train)
        Y_train_pred = model.predict_classes(X_train_boosting).tolist()
        Y_test_pred = model.predict_classes(X_test_boosting).tolist()
        if i != n_models:
            new_X_train_boosting = np.concatenate(Y_train_pred, X_train[n_models_together:], axis=1)
            new_X_test_boosting = np.concatenate(Y_test_pred, X_test[n_models_together:], axis=1)

            X_train_boosting = new_X_train_boosting[:, :n_models_together]
            X_test_boosting = new_X_test_boosting[:, :n_models_together]

    return Y_test_pred