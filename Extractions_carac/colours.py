"""Test the colour feature extraction."""
from PIL import Image
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv1D, GlobalAveragePooling1D
from keras.layers import Dense
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint

import showing_infos_Windstorm
import accuracies_metrics

def get_colours(img1):
    img1 = Image.fromarray(img1, 'RGB')
    w, h = img1.size
    colour_list = img1.getcolors(w * h)
    return sorted(colour_list, key=lambda x: x[0], reverse=True)


def get_relevant_colours(list_cols_freq):
    most_frequent_colours = [
        elt for elt in list_cols_freq if elt[1] not in [(0, 0, 0), (65, 0, 0)]]
    most_frequent_colours = most_frequent_colours[:5]
    most_frequent_colours = [[elt[0], *elt[1]]
                             for elt in most_frequent_colours]
    l_toadd = [[0, -1, -1, -1]] * (5 - len(most_frequent_colours))
    most_frequent_colours += l_toadd
    if len(l_toadd) != 0:
        print(most_frequent_colours)
    return most_frequent_colours


def arrange_colour_list(list_colours, error, n_pixels):
    def round_to_nearest_multiple(n, error):
        return round(n / error) * error

    colours = []
    for elt in list_colours:
        colours.append([elt[0] * (100 / n_pixels), (round_to_nearest_multiple(elt[1][0], error),
                                                    round_to_nearest_multiple(elt[1][1], error),
                                                    round_to_nearest_multiple(elt[1][2], error))])

    new_coulour_list = []
    frequence_list = []
    for frequence, colour in colours:
        if colour in new_coulour_list:
            frequence_list[new_coulour_list.index(colour)] += frequence
        else:
            new_coulour_list.append(colour)
            frequence_list.append(frequence)

    res = []
    for i in range(len(frequence_list)):
        res.append((frequence_list[i], new_coulour_list[i]))
    return res


# We extract the 5 more commun colours of the  group of frames.


def get_colour_features(X_, frame_shape):
    colFeatures = []
    for group_of_frames in X_:
        colours = get_colours(group_of_frames[0])

        colAndfreqs = arrange_colour_list(
            colours, 5, frame_shape[0] * frame_shape[1])
        colAndfreqs = get_relevant_colours(colAndfreqs)
        colFeatures.append(colAndfreqs)
        if len(colAndfreqs) != len(colFeatures[-1]):
            print("error in lengths : ", colAndfreqs)
    return colFeatures


def create_model(learning_rate, n_classes):
    model = Sequential()
    model.add(Conv1D(1000, 3, activation='relu', input_shape=(5, 4)))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(n_classes, activation='sigmoid'))

    optimizer = Adam(lr=learning_rate)  #
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.load_weights("Windstorm/Models/colours.hdf5")

    return model


def fitting_colour_model(n_classes, X_train_colours, X_test_colours, Y_train,
                         Y_test):
    """Create and fit model to be trained on colours.

    Currently, does not give good results at all.
    """
    checkpointer_colours = ModelCheckpoint(filepath='../Models/colours.hdf5', verbose=1, save_best_only=True)
    CNN_colours = create_model(learning_rate=5 * 10 ** -6, n_classes=n_classes)
    print("fitting on colour model")
    array_test_colours = np.array(X_test_colours)
    history = CNN_colours.fit(np.array(X_train_colours), Y_train,
                              batch_size=400,
                              validation_data=(array_test_colours, Y_test),
                              epochs=1000, verbose=1, shuffle=True,
                              callbacks=[checkpointer_colours, EarlyStopping(patience=100)])
    showing_infos_Windstorm.show_accuracy_over_time(history, "colour model")
    Y_test_pred_CNN_colours = CNN_colours.predict(np.array(X_test_colours))
    print(accuracies_metrics.mat_conf(Y_test, Y_test_pred_CNN_colours))
    CNN_colours.load_weights("../Models/colours.hdf5")
    Y_train_pred_CNN_colours = CNN_colours.predict(np.array(X_train_colours))
    return Y_train_pred_CNN_colours
