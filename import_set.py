"""Here functions concerning data import are stored."""

import pickle as pkl
import data_processing
import numpy as np

import video_processing

import LSTMwithKeras
from Extractions_carac import diff_between_frames
from Extractions_carac import colours
from sklearn.preprocessing import label_binarize


def get_labels(video_names):
    """Labels videos, according to their name.

    As labels we will use the beaufort scale reduced.
    The beaufort scale is labelled from 0 to 12
    0 - 1 - 2 in the beaufort scale correspond to really calm winds,
       nearly nothing, and then, to 0 in our scale
    3 - 4 - 5 correspond to calm wind -> 1 in our scale
    6 to 7 in the beaufort scale correspond to high winds and then, to 2
    8 to 9 correspond to gale force -> 3
    10 to 11 correspond to storm -> 4
    12 correspond to hurricane -> 5
    """
    labels = []
    for name in video_names:
        name_without_type = name[:name.index('.')]
        labels.append(int(name_without_type[-1]))

    return labels


class Data_sets:
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = None
        self.Y_test = None


def make_Y(labels, n_groups_of_frames_per_video):
    """Make Y set with the labels.

    Labels are labels corresponding to the videos. 
    Here we want to have a labelling corresponding to each subvideo ( group of frame).
    Thus we take labels and we copy each label by n , 
    with n = the number of subvideos corresponding to the video labelled.
    """

    return sum(([label] * n_groups_of_frame for label, n_groups_of_frame in
                zip(labels, n_groups_of_frames_per_video)),
               [])


def get_Y(names, n_groups_of_frames_per_video, n_classes):
    """Return Y set formated."""
    labels = get_labels(names)
    Y = make_Y(labels, n_groups_of_frames_per_video)
    Y = label_binarize(Y, classes=[i for i in range(n_classes)])
    return Y


def get_features_from_one_video(video_name, names):
    """Return features from one video."""
    print("video : ", video_name, ", number ", 1 +
          names.index(video_name), "for ", len(names))
    video, n_frames_per_video = video_processing.get_a_video_as_array(
        size=(299, 299), prefix="../Videos/", frames_per_sample=15,
        video_name=video_name)

    X_features_CNN_loc = LSTMwithKeras.set_to_features(video)
    X_features_pixelDifference_loc = diff_between_frames.move_extraction(
        video, frames_per_sample=15)
    X_features_colours_loc = colours.get_colour_features(
        video, frame_shape=(299, 299))

    print("len of X_features_CNN_loc ", len(X_features_CNN_loc))
    print("len of X_features_pixelDifference_loc ",
          len(X_features_pixelDifference_loc))
    print("len of X_features_colours_loc ", len(X_features_colours_loc))

    print("-------------------------------------------------")
    print("len of X_features_CNN ", len(X_features_CNN))
    print("len of X_features_pixelDifference",
          len(X_features_pixelDifference))
    print("len of X_features_colours", len(X_features_colours))
    print("len of n_subvideos_per_video ", len(n_subvideos_per_video))

    return X_features_CNN_loc, list(
        X_features_pixelDifference_loc), X_features_colours_loc, n_frames_per_video


def get_features(names, n_classes=3):
    """Return features of different extractions done here."""
    X_features_CNN = []
    X_features_pixelDifference = []
    X_features_colours = []
    n_subvideos_per_video = []

    for video_name in names:
        X_features_CNN_loc, X_features_pixelDifference_loc, X_features_colours_loc, n_frames_per_video = get_features_from_one_video(
            video_name, names)

        X_features_CNN += X_features_CNN_loc
        X_features_pixelDifference += list(X_features_pixelDifference_loc)
        X_features_colours += X_features_colours_loc
        n_subvideos_per_video.append(n_frames_per_video)

    Y = get_Y(names, n_subvideos_per_video, n_classes)
    groups_frames_names = []
    for i in range(len(names)):
        for j in range(n_subvideos_per_video[i]):
            groups_frames_names.append(names[i])

    return (X_features_CNN, X_features_pixelDifference, X_features_colours,
            n_subvideos_per_video, Y, groups_frames_names)


def return_extracted_set(X_features_CNN, X_features_pixelDifference,
                         X_features_colours, n_subvideos_per_video):
    print("--------------------- Final Set ----------------------")
    print("len of X_features_CNN ", len(X_features_CNN))
    print("len of X_features_pixelDifference",
          len(X_features_pixelDifference))
    print("len of X_features_colours", len(X_features_colours))
    print("len of n_subvideos_per_video ", len(n_subvideos_per_video))
    return X_features_CNN, X_features_pixelDifference, X_features_colours, n_subvideos_per_video, Y, groups_frames_names


def combine_previous_sets_with_recent(feature_result, X_features_CNN,
                                      X_features_pixelDifference,
                                      X_features_colours, n_subvideos_per_video,
                                      Y, groups_frames_names):
    """Add calculated features to the old one."""
    X_features_CNN_new = feature_result[0]
    X_features_pixelDifference_new = feature_result[1]
    X_features_colours_new = feature_result[2]
    n_subvideos_per_video_new = feature_result[3]
    Y_new = feature_result[4]
    groups_frames_names_new = feature_result[5]

    X_features_CNN += X_features_CNN_new

    X_features_pixelDifference += X_features_pixelDifference_new

    X_features_colours += X_features_colours_new
    print(type(X_features_colours[0]))
    n_subvideos_per_video += n_subvideos_per_video_new

    Y = list(Y) + list(Y_new)
    Y = np.array(Y)

    groups_frames_names += groups_frames_names_new
    return X_features_CNN, X_features_pixelDifference, X_features_colours, n_subvideos_per_video, Y, groups_frames_names


def get_previous_set(path_to_pickle):
    """Try to get previous features, else, return empty files."""

    try:
        Set = pkl.load(
            open(path_to_pickle + "/3TypesOfFeatures.pkl", "rb"))
        X_features_CNN = Set[0]
        X_features_pixelDifference = Set[1]
        X_features_colours = Set[2]
        n_subvideos_per_video = Set[3]
        Y = Set[4]
        groups_frames_names = Set[5]

    except:
        X_features_CNN, X_features_pixelDifference = [], []
        X_features_colours, n_subvideos_per_video = [], []
        Y, groups_frames_names = [], []

    return X_features_CNN, X_features_pixelDifference, X_features_colours, n_subvideos_per_video, Y, groups_frames_names


def get_set(load_pickle, parameters):
    """Return train and test set of all different feature extractions."""
    path_to_pickle = "../Pickles/3TypesOfFeatures.pkl"  # data_processing.get_path("Pickles", "3TypesOfFeatures.pkl")
    names_wanted = parameters.names_wanted
    if load_pickle:
        Set = pkl.load(open(path_to_pickle, "rb"))
        res = (Set[0], Set[1], Set[2], Set[3], Set[4], Set[5])
        return res

    else:
        X_features_CNN, X_features_pixelDifference, X_features_colours, n_subvideos_per_video, Y, groups_frames_names = get_previous_set(
            path_to_pickle)

        if names_wanted == "all":
            all_names = data_processing.get_all_names_from_path("Windstorm/videos")
        else:
            all_names = names_wanted

        names_already_pickled = list(set(groups_frames_names))

        names_to_pickle = list(set(all_names) - set(names_already_pickled))

        if len(names_to_pickle) == 0:
            return return_extracted_set(X_features_CNN,
                                        X_features_pixelDifference,
                                        X_features_colours,
                                        n_subvideos_per_video)

        X_features_CNN, X_features_pixelDifference, X_features_colours, n_subvideos_per_video, Y, groups_frames_names = get_features(
            names_to_pickle)

        pkl.dump((X_features_CNN, X_features_pixelDifference,
                  X_features_colours, n_subvideos_per_video,
                  Y, groups_frames_names),
                 open(path_to_pickle, "wb"))

        return_extracted_set(X_features_CNN, X_features_pixelDifference,
                             X_features_colours, n_subvideos_per_video)


def make_train_test_set(X_features_CNN, X_features_pixelDifference,
                        X_features_colours, n_subvideos_per_video, Y,
                        groups_frames_names, percentage_testset,
                        parameters):
    """Return all train and test set.

    We want to respect two constraints :
    -All subvideos from a video are on the same set
    -The proportions of videos of the differents classes are equal on
    the train set
    """
    names = list(set(groups_frames_names))
    labels = get_labels(names)
    names_train_imposed = parameters.names_train
    indexs_1 = []
    indexs_0 = []
    for i in range(len(labels)):
        if labels[i] == 1:
            indexs_1.append(i)
        else:
            indexs_0.append(i)

    nb_classes = 2
    nb_video_train = int(len(names) * (1 - percentage_testset) / nb_classes)

    index_train = indexs_1[0:nb_video_train] + indexs_0[0:nb_video_train]
    index_test = list(set(list(range(len(labels)))) - set(index_train))

    names_train = [names[i] for i in index_train]
    names_test = [names[i] for i in index_test]

    if names_train_imposed is not None:
        names_train = names_train_imposed

    print(" names of train set videos : ", names_train)
    print(" names of test set videos : ", names_test)

    def train_test_attribution(Set, names_train, groups_frames_names):
        train, test = [], []
        for i in range(len(Set)):
            if groups_frames_names[i] in names_train:
                train.append(Set[i])
            else:
                test.append(Set[i])

        return train, test

    cnn_features_extracted = Data_sets(*train_test_attribution(X_features_CNN, names_train, groups_frames_names))
    pixel_difference = Data_sets(*train_test_attribution(X_features_pixelDifference, names_train, groups_frames_names))
    colours = Data_sets(*train_test_attribution(X_features_colours, names_train, groups_frames_names))

    Y_train, Y_test = train_test_attribution(
        Y, names_train, groups_frames_names)
    Y_train, Y_test = np.array(Y_train), np.array(Y_test)

    groups_frames_names_train, groups_frames_names_test = train_test_attribution(
        groups_frames_names, names_train, groups_frames_names)

    return cnn_features_extracted, pixel_difference, colours, Y_train, Y_test, groups_frames_names_train, groups_frames_names_test


def get_assemble_set(preds):
    """Make a set for voting, given prediction done by the models."""
    def create_one_assemble(preds):
        X_assemble=[]
        for i in range(len(preds[0])):
            sublist = []
            for j in range(len(preds)):
                sublist += preds[j][i].tolist()
            X_assemble.append(sublist)
        return X_assemble

    return Data_sets(create_one_assemble(preds.X_train), create_one_assemble(preds.X_test))

def get_names_train():
    """Return a list of names_train."""
    with open('info_files/names_train.txt') as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    return content
