from Windstorm_Classification.Data_Managing import importing_data
from Windstorm_Classification.ML_Models import raw_CNN_LSTM
from Windstorm_Classification.ML_Models import pretrained_CNN_LSTM
from Windstorm_Classification.ML_Models import capsules_LSTM
from Windstorm_Classification.ML_Models.all_in_models import Conv3d

import numpy as np
import pickle as pkl

frame_rate_wanted = 8
n_frame_per_subvideo = 50
shape_wanted = (180, 180)
np.random.seed(42)

names, labels, videos = importing_data.import_set('../Data/videos/',
                                                  frame_rate_wanted,
                                                  n_frame_per_subvideo,
                                                  shape_wanted)
dataset_dico = importing_data.get_set(names,
                                      labels,
                                      videos,
                                      test_percentage=0.1,
                                      val_percentage=0.4)
model_wanted = 'capsule_lstm'

if model_wanted == 'raw_CNN':
    model = raw_CNN_LSTM.cnn_lstm(shape=(n_frame_per_subvideo, shape_wanted[0], shape_wanted[1], 3),
                                  n_classes=2,
                                  learning_rate=10 ** -3,
                                  decay=0.01)
if model_wanted == "pretrained_CNN":
    model = pretrained_CNN_LSTM.lstm(shape=(n_frame_per_subvideo, 2048),
                                     n_classes=2,
                                     learning_rate=10 ** -3,
                                     decay=0.01)

    try:
        dataset_dico = pkl.load(open("../Data/dataset_dico_pretrained_CNN", "rb"))
    except:
        dataset_dico['X_train'] = np.array(pretrained_CNN_LSTM.set_to_features(dataset_dico['X_train']))
        dataset_dico['X_val'] = np.array(pretrained_CNN_LSTM.set_to_features(dataset_dico['X_val']))
        dataset_dico['X_test'] = np.array(pretrained_CNN_LSTM.set_to_features(dataset_dico['X_test']))

        pkl.dump(dataset_dico, open("../Data/dataset_dico_pretrained_CNN", "wb"))

if model_wanted == "capsule_lstm":
    model = capsules_LSTM.capsules_lstm(shape=(n_frame_per_subvideo, shape_wanted[0], shape_wanted[1], 3),
                                        n_classes=2,
                                        learning_rate=10 ** -3,
                                        decay=0.01)

if model_wanted == "conv3d":
    model = Conv3d.c3d(input_shape=(n_frame_per_subvideo, shape_wanted[0], shape_wanted[1], 3),
                       nb_classes=2,
                       learning_rate=10 ** -3,
                       decay=0.01)

model.fit(dataset_dico['X_train'],
          dataset_dico['Y_train'],
          verbose=1,
          batch_size=10,
          validation_data=(dataset_dico['X_val'], dataset_dico['Y_val']),
          epochs=10)

print(model.predict(dataset_dico['X_test']))
print(set(dataset_dico['names_test']))
