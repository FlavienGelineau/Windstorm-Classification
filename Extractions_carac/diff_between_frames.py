"""Test a feature extraction based on frame differences."""
import data_processing
from PIL import Image
import numpy as np

import pickle as pkl
from keras.optimizers import Adam

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, GlobalAveragePooling1D

import video_processing

from keras.layers import Activation, LSTM
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint

import showing_infos_Windstorm
import accuracies_metrics


def coulours_to_bw(data):
    img = Image.fromarray(data, 'RGB')
    gray = img.convert('L')
    bw = gray.point(lambda x: 0 if x < 128 else 255, '1')
    bw = np.asarray(bw, dtype="int32")

    return bw


def diff_between(image1, image2):
    diff = np.absolute(image1 ^ image2)

    return np.sum(diff), diff.size


def move_extraction(X, frames_per_sample):
    differences = []
    for i in range(len(X)):

        diff_1_video = [0] * (frames_per_sample - 1)
        for j in range(len(X[i]) - 1):
            bw1 = coulours_to_bw(X[i][j])
            bw2 = coulours_to_bw(X[i][j + 1])
            sum_diff, total = diff_between(bw1, bw2)
            diff_1_video[j] = sum_diff / total

        differences.append(diff_1_video)

    return np.expand_dims(differences, axis=2)


def create_model(learning_rate, n_classes):

    model = Sequential()
    model.add(Conv1D(100, 3, activation='relu', input_shape=(14, 1)))
    model.add(Conv1D(60, 3, activation='relu'))
    model.add(Conv1D(30, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv1D(20, 3, activation='relu'))
    model.add(Conv1D(15, 2, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(n_classes, activation='sigmoid'))

    optimizer = Adam(lr=learning_rate)  #
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.load_weights("Windstorm/Models/pixel_diff.hdf5")

    return model


def picture_diff(array1, array2):
    res = np.zeros((array1.shape[0], array1.shape[1], 1))

    for i in range(len(array1)):
        for j in range(len(array1[i])):
            if np.array_equal(array1[i][j], array2[i][j]) == False:
                res[i][j][0] = 1
    return res


def move_extraction_image(X):
    res = []
    for i in range(len(X)):  # For each group of frame / subvideo of the video
        gp_img = []
        # for each couple of frames in the group of frames
        for j in range(len(X[i]) - 1):
            gp_img.append(picture_diff(X[i][j], X[i][j + 1]))
        res.append(np.array(gp_img))
        # imgplot = plt.imshow(res)
        # plt.show()
    return res


def get_data_set_img_difference(video_names):
    data_set = []
    n_subvideos_per_video = []
    n_classes = 3
    for name in video_names:
        print(name)
        video, n_groups_of_frames_per_video = video_processing.getOneVideoasarray(
            size=(90, 90), prefix="Windstorm/videos/", frames_per_sample=15, video_name=name)
        res = move_extraction_image(video)
        data_set += res

        n_subvideos_per_video.append(n_groups_of_frames_per_video)

    Y = getSets.getY(video_names, n_subvideos_per_video,  n_classes)

    return np.array(data_set), Y


def crnn(X_train):
    """Build a CNN into RNN.
    Starting version from:
    https://github.com/udacity/self-driving-car/blob/master/
        steering-models/community-models/chauffeur/models.py
    """
    model = Sequential()
    model.add(TimeDistributed(Conv2D(80, (8, 8),
                                     kernel_initializer="he_normal",
                                     activation='relu'), input_shape=(14, 90, 90, 1)))
    model.add(TimeDistributed(Conv2D(60, (6, 6),
                                     kernel_initializer="he_normal",
                                     activation='relu')))
    model.add(TimeDistributed(MaxPooling2D()))
    model.add(TimeDistributed(Conv2D(48, (5, 5),
                                     kernel_initializer="he_normal",
                                     activation='relu')))
    model.add(TimeDistributed(Conv2D(48, (3, 3),
                                     kernel_initializer="he_normal",
                                     activation='relu')))
    model.add(TimeDistributed(MaxPooling2D()))
    model.add(TimeDistributed(Conv2D(64, (3, 3),
                                     kernel_initializer="he_normal",
                                     activation='relu')))
    model.add(TimeDistributed(Conv2D(64, (3, 3),
                                     kernel_initializer="he_normal",
                                     activation='relu')))
    model.add(TimeDistributed(MaxPooling2D()))
    model.add(TimeDistributed(Conv2D(128, (3, 3),
                                     kernel_initializer="he_normal",
                                     activation='relu')))
    model.add(TimeDistributed(MaxPooling2D()))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(256, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    optimizer = Adam(lr=10**-7)  # aggressively small learning rate
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def CNN_LSTM_from_scratch(X_train):

    model = Sequential()

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same'),
                              input_shape=(14, 80, 80, 1)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv2D(32, (3, 3))))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    # model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(20)))

    model.add(TimeDistributed(Dense(35, name="first_dense")))

    model.add(LSTM(20, return_sequences=True, name="lstm_layer"))

    #%%
    model.add(Flatten())
    model.add(Dense(3, activation='relu'))

    #%%

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model



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
    CNN = create_model(learning_rate=10 ** -7, n_classes=n_classes)
    print("fitting on pixel difference model")
    X_train = np.array(X_train_pixelDifference)
    validation_data = (np.array(X_test_pixelDifference), np.array(Y_test))
    history = CNN.fit(X_train, Y_train, validation_data=validation_data,
                      batch_size=400, epochs=1000, verbose=1, shuffle=True,
                      callbacks=[checkpointer_pixel_diff, EarlyStopping(patience=100)])
    showing_infos_Windstorm.show_accuracy_over_time(history, "pixel difference model")
    Y_test_pred_CNN_pixel_diff = CNN.predict(np.array(X_test_pixelDifference))
    accuracies_metrics.mat_conf(Y_test, Y_test_pred_CNN_pixel_diff)

    CNN.load_weights("../Models/pixel_diff.hdf5")
    Y_train_pred_CNN_pixel_diff = CNN.predict(X_train)
    return Y_train_pred_CNN_pixel_diff


def generate_arrays_from_file(all_names, batch_size):

    while 1:
        nb_batch = (len(X) // batch_size) + 1
        for i in range(nb_batch):
            #print (np.array(X[i]),Y[i])
            names = all_names[i:i + batch_size]
            X_set, Y_set = get_data_set_img_difference(names)
            X_set, Y_set = shuffle(X_set, Y_set)
            yield (X_set, Y_set)


from sklearn.utils import shuffle

if __name__ == '__main__':
    all_names = data_processing.getAllNamesFromPath("Windstorm/videos")

    names_train = ['80-0.mp4', '280-0.mp4', '184-0.mp4', '180-1.mp4', '126-1.mp4', '143-1.mp4', '164-1.mp4', '188-1.mp4', '133-1.mp4', '200-1.mp4', '111-1.mp4', '141-1.mp4', '105-1.mp4', '22-1.mp4', '146-1.mp4', '168-1.mp4', '136-1.mp4',
                   '178-1.mp4', '172-1.mp4', '170-1.mp4', '185-1.mp4', '101-1.mp4', '118-1.mp4', '20-1.mp4', '104-1.mp4', '195-1.mp4', '30-1.mp4', '123-1.mp4', '100-1.mp4', '112-1.mp4', '175-1.mp4', '157-1.mp4', '127-1.mp4', '27-1.mp4', '35-1.mp4',
                   '191-1.mp4', '150-1.mp4', '24-1.mp4', '107-1.mp4', '93-1.mp4', '129-1.mp4', '37-1.mp4', '155-1.mp4', '14-1.mp4', '26-1.mp4', '90-1.mp4', '33-1.mp4', '158-1.mp4', '262-1.mp4', '116-1.mp4', '176-1.mp4', '194-1.mp4', '19-1.mp4',
                   '159-1.mp4', '198-1.mp4', '177-1.mp4', '110-1.mp4', '242-1.mp4', '89-1.mp4', '144-1.mp4', '115-1.mp4', '87-1.mp4', '163-1.mp4', '131-1.mp4', '134-1.mp4', '103-1.mp4', '204-1.mp4', '92-1.mp4', '138-1.mp4', '182-1.mp4', '16-1.mp4',
                   '291-0.mp4', '207-0.mp4', '279-0.mp4', '276-0.mp4', '55-0.mp4', '12-1.mp4', '91-1.mp4', '99-1.mp4', '61-1.mp4', '135-1.mp4', '183-1.mp4', '169-1.mp4', '124-1.mp4', '28-1.mp4', '192-1.mp4', '179-1.mp4', '109-1.mp4', '203-1.mp4',
                   '34-1.mp4', '88-1.mp4', '202-1.mp4', '147-1.mp4', '122-1.mp4', '128-1.mp4', '21-1.mp4', '189-1.mp4', '161-1.mp4', '205-1.mp4', '114-1.mp4', '13-1.mp4', '117-1.mp4', '94-1.mp4', '149-1.mp4', '121-1.mp4', '98-1.mp4', '167-1.mp4',
                   '18-1.mp4', '142-1.mp4', '187-1.mp4', '108-1.mp4', '171-1.mp4', '132-1.mp4', '120-1.mp4', '153-1.mp4', '125-1.mp4', '210-0.mp4', '77-0.mp4', '58-0.mp4', '288-0.mp4', '250-0.mp4', '213-0.mp4', '74-0.mp4', '231-2.mp4', '6-2.mp4',
                   '130-2.mp4', '246-2.mp4', '83-2.mp4', '230-2.mp4', '240-2.mp4', '41-0.mp4', '75-0.mp4', '237-0.mp4', '71-0.mp4', '235-0.mp4', '233-2.mp4', '253-2.mp4', '31-0.mp4', '221-2.mp4', '79-0.mp4', '201-2.mp4', '247-0.mp4', '43-0.mp4',
                   '269-0.mp4', '269-0.mov', '67-0.mp4', '50-0.mp4', '76-0.mp4', '292-0.mp4', '46-0.mp4', '212-0.mp4', '214-0.mp4', '264-0.mp4', '42-0.mp4', '244-2.mp4', '238-0.mp4', '70-2.mp4', '11-0.mp4', '48-0.mp4', '5-2.mp4', '1-0.mp4',
                   '78-0.mp4', '279-0.mp4', '44-0.mp4', '57-0.mp4', '140-0.mp4', '39-0.mp4', '252-0.mp4', '232-2.mp4', '284-0.mp4', '145-2.mp4', '243-0.mp4', '40-0.mp4', '227-0.mp4', '216-0.mp4', '272-0.mp4', '277-0.mp4', '52-0.mp4', '7-2.mp4',
                   '285-2.mp4', '209-0.mp4', '165-2.mp4', '283-2.mp4', '56-0.mp4', '226-2.mp4', '239-2.mp4', '85-2.mp4', '220-0.mp4', '290-2.mp4', '215-0.mp4', '225-0.mp4', '47-0.mp4', '206-0.mp4', '160-0.mp4', '62-0.mp4', '229-2.mp4', '236-0.mp4',
                   '154-2.mp4', '66-0.mp4', '259-2.mp4', '278-0.mp4', '64-0.mp4', '219-0.mp4', '228-0.mp4', '245-2.mp4', '266-0.mp4', '55-0.mp4', '53-0.mp4', '3-0.mp4', '199-2.mp4', '51-0.mp4', '4-2.mp4', '80-0.mp4', '270-0.mp4', '49-0.mp4',
                   '249-0.mp4', '65-0.mp4', '267-0.mp4', '218-0.mp4', '274-0.mp4', '217-0.mp4', '173-2.mp4', '59-0.mp4', '224-0.mp4', '139-2.mp4', '73-0.mp4', '286-2.mp4', '81-2.mp4', '8-2.mp4', '291-0.mp4', '69-0.mp4', '275-0.mp4',
                   '148-2.mp4', '151-2.mp4', '297-2.mp4', '294-2.mp4', '295-2.mp4', '256-2.mp4', '152-2.mp4', '32-2.mp4', '258-2.mp4'
                   ]
    names_test = list(set(all_names) - set(names_train))
    print(names_test)

    get_pickle = True
    if get_pickle:
        Set = pkl.load(
            open("Windstorm/Pickles/image_difference_set.pkl", "rb"))
        X_train, Y_train, X_test, Y_test = Set[0], Set[1], Set[2], Set[3]
    else:

        X_test, Y_test = get_data_set_img_difference(names_test)

        X_train, Y_train = get_data_set_img_difference(names_train)
        pkl.dump((X_train, Y_train, X_test, Y_test), open(
            "Windstorm/Pickles/image_difference_set.pkl", "wb"))

    model = crnn(X_train)
    batch_size = 80
    #model.fit_generator(generator = generate_arrays_from_file(names_train, batch_size), validation_data = (X_test, Y_test), steps_per_epoch = 2*int(len(Y_train)/batch_size), epochs=30, verbose=1)

    model.fit(X_train, Y_train, validation_data=(
        X_test[:100], Y_test[:100]), verbose=True, batch_size=150, epochs=300)
    Y_pred = model.predict(X_test)

    mat_conf(Y_test, Y_pred)
