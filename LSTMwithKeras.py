"""Function around the LSTM used in this model."""
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
import numpy as np

from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential

from keras.layers.recurrent import LSTM
from keras.optimizers import Adam


class Extractor():
    """Create an Extractor object which will extract spatial features."""

    def __init__(self, weights=None):
        """Load pretrained from imagenet."""
        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            # Get model with pretrained weights.
            base_model = InceptionV3(
                weights='imagenet',
                include_top=True
            )

            # We'll extract features at the final pool layer.
            self.model = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('avg_pool').output
            )

        else:
            # Load the model first.
            self.model = load_model(weights)

            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

    def extract(self, x):
        """Return features."""
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features


def video_to_features(vid):
    """Return features for every frame of the video."""
    ext = Extractor()
    return [ext.extract(vid[j]) for j in range(len(vid))]


def set_to_features(X_set):
    """Return features from set."""
    ext = Extractor()
    features = []
    for i in range(len(X_set)):
        print(i, " out of ", len(X_set))
        bag_of_features = []
        for j in range(len(X_set[i])):
            bag_of_features.append(ext.extract(X_set[i][j]))
        features.append(bag_of_features)

    return features


def lstm(shape, n_classes, learning_rate, decay):
    """Build a simple LSTM network.

    We pass the extracted features from our CNN to this model predomenently.
    """
    # Model.
    model = Sequential()
    model.add(LSTM(4000, return_sequences=True, input_shape=shape, dropout=0.5))
    model.add(Flatten())
    model.add(Dense(700, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax'))

    # aggressively small learning rate
    optimizer = Adam(lr=learning_rate, decay=decay)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    # model.save_weights("Windstorm/Models/Initial_weights_LSTM_CNN.hdf5")
    # model.load_weights("Windstorm/Models/LSTM_CNN.hdf5")
    return model
