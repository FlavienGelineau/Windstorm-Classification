from keras.layers import Dense, Flatten, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adam

from Windstorm_Classification.ML_Models.Capsules import *


def capsules_lstm(shape, n_classes, learning_rate, decay):
    model = Sequential()
    caps = CapsNet(input_shape=shape[1:],
                   n_class=2,
                   routings=2)
    model.add(TimeDistributed(caps, input_shape = shape))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128, return_sequences=False, dropout=0.5))
    model.add(Dense(n_classes, activation='softmax'))

    optimizer = Adam(lr=learning_rate, decay=decay)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())
    return model
