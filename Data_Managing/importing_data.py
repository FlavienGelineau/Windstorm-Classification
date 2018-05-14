import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from skvideo.io import vread, ffprobe, vwrite
import gc
from keras.utils.np_utils import to_categorical

from Windstorm_Classification.Data_Managing import processing_data
from Windstorm_Classification.Data_Managing import data_augmentation


def get_all_names_from_path(path):
    """Return the names of all the files from a given path.

    Inputs : path, string
    Outputs : a list of strings, which are the names of all files in the path
    """
    return [f for f in listdir(path) if isfile(join(path, f))]


def get_frame_rate_from_video(name):
    metadata = ffprobe(name)
    metadatas = dict(metadata['video'])

    frame_rate = metadatas['@avg_frame_rate'].split('/')
    return int(int(frame_rate[0]) / int(frame_rate[1]))


def get_video(base_path, name, frame_rate_wanted, shape_wanted):
    if name in get_all_names_from_path(base_path + "videos_reduced/"):
        return vread(base_path + "videos_reduced/{}".format(name))
    video = vread(base_path + name)
    frame_rate = get_frame_rate_from_video(base_path + name)
    print('video imported')
    video = processing_data.process_frame_rate(video, frame_rate, frame_rate_wanted)
    video = processing_data.process_frame_shape(video, shape_wanted)
    gc.collect()
    vwrite(base_path + "videos_reduced/{}".format(name), video)
    return video


def get_label_from_name(name):
    return [int(name.split('_')[0])]


def import_set(base_path, frame_rate_wanted, threshold, shape_wanted):
    names = get_all_names_from_path(base_path)
    labels = [get_label_from_name(name) for name in names]
    videos = [get_video(base_path, name, frame_rate_wanted, shape_wanted) for name in names]

    videos, names, labels = data_augmentation.cut_long_videos(names, labels, videos, threshold)
    labels = to_categorical(labels, num_classes=2)
    return names, labels, videos


def get_train_test_set(names, labels, videos, test_percentage):
    unique_names = list(set(names))
    limit_train = int(len(unique_names) * test_percentage)

    names_train = unique_names[limit_train:]
    names_test = unique_names[:limit_train]

    labels_train, labels_test = [], []
    videos_train, videos_test = [], []

    for name, label, video in zip(names, labels, videos):
        if name in names_train:
            labels_train.append(label)
            videos_train.append(video)
        else:
            labels_test.append(label)
            videos_test.append(video)

    names_train = [name for name in names if name in names_train]
    names_test = [name for name in names if name in names_test]
    return names_train, names_test, labels_train, labels_test, np.array(videos_train), np.array(videos_test)

def get_set(names, labels, videos, test_percentage, val_percentage):
    names_train, names_test, Y_train, Y_test, X_train, X_test = get_train_test_set(
        names, labels, videos, test_percentage=test_percentage)

    names_train, names_val, Y_train, Y_val, X_train, X_val = get_train_test_set(
        names_train, Y_train, X_train, test_percentage=val_percentage)

    return {"names_train":names_train,
            "names_val":names_val,
            "names_test":names_test,
            "Y_train":np.array(Y_train),
            "Y_val":np.array(Y_val),
            "Y_test": np.array(Y_test),
            "X_train":np.array(X_train),
            "X_val":np.array(X_val),
            "X_test":np.array(X_test)}