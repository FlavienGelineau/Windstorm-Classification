from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import (MaxPooling3D, Conv3D)
from keras.models import Sequential
from keras.optimizers import Adam


def c3d(input_shape, nb_classes, learning_rate, decay):
    """
    Build a 3D convolutional network, aka C3D.
        https://arxiv.org/pdf/1412.0767.pdf
    With thanks:
        https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2
    """
    model = Sequential()
    # 1st layer group
    model.add(Conv3D(64, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv1',
                     subsample=(1, 1, 1),
                     input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv2',
                     subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv3a',
                     subsample=(1, 1, 1)))
    model.add(Conv3D(256, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv3b',
                     subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3'))

    model.add(Conv3D(512, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv4b',
                     subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool4'))

    model.add(Flatten())

    # FC layers group
    model.add(Dense(1024, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))


    optimizer = Adam(lr=learning_rate, decay=decay)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    return model
