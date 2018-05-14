from keras.models import Model, load_model
import numpy as np

from keras.layers import Dense, Flatten, Dropout, TimeDistributed, Conv2D, MaxPool2D
from keras.models import Sequential

from keras.layers.recurrent import LSTM
from keras.optimizers import Adam


def cnn_lstm(shape, n_classes, learning_rate, decay):
    """Build a CNN into RNN.
    Starting version from:
        https://github.com/udacity/self-driving-car/blob/master/
            steering-models/community-models/chauffeur/models.py
    Heavily influenced by VGG-16:
        https://arxiv.org/abs/1409.1556
    Also known as an LRCN:
        https://arxiv.org/pdf/1411.4389.pdf
    """
    model = Sequential()

    model.add(TimeDistributed(Conv2D(32, (8, 8), strides=(2, 2),
                                     activation='relu', padding='same'), input_shape=shape))
    model.add(TimeDistributed(Conv2D(32, (5, 5),
                                     kernel_initializer="he_normal", activation='relu')))
    model.add(TimeDistributed(MaxPool2D((5, 5), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(64, (5, 5),
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(64, (5, 5),
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPool2D((5, 5), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(128, (5, 5),
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(128, (5, 5),
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPool2D((5, 5), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(256, (3, 3),
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(256, (3, 3),
                                     padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPool2D((3, 3), strides=(2, 2))))

    model.add(TimeDistributed(Flatten()))

    model.add(Dropout(0.5))
    model.add(LSTM(256, return_sequences=False, dropout=0.5))
    model.add(Dense(n_classes, activation='softmax'))

    optimizer = Adam(lr=learning_rate, decay=decay)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    return model
