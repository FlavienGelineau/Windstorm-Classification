"""All functions about accuracy."""
import data_processing
from sklearn.metrics import confusion_matrix


def percentage_accuracy_videos(Y_pred, Y_test, groups_frames_names):
    """Give accuracy on each video, given their name."""
    names = list(set(groups_frames_names))
    percentage = [0] * len(names)
    acc = dict(zip(names, percentage))

    for pred, y_test_element, group_frame_name in zip(Y_pred, Y_test, groups_frames_names):
        list_pred_i = pred.tolist()
        if list_pred_i.index(max(pred)) == y_test_element.tolist().index(1):
            acc[group_frame_name] += 1

    for key, values in acc.items():
        acc[key] = float(values) / float(groups_frames_names.count(key))
    return acc


def handmade_valacc(Y_pred, Y_test):
    """Recalculates handmade acc to be sure of value given."""
    acc = 0
    for pred in Y_pred:
        list_pred_i = pred.tolist()
        if list_pred_i.index(max(pred)) == list_pred_i.index(1):
            acc += 1
    return acc / len(Y_pred)


def mat_conf(Y_test, Y_pred, name=""):
    """Show confusion matrix and accuracy on test set."""
    if type(Y_pred[0])!=int: # Thus Y_pred[0] is an array or a list
        if len(Y_pred[0])>1:
            Y_pred = data_processing.max_in_each_list(Y_pred)
        else:
            Y_pred = [pred[0] for pred in Y_pred]
    Y_test_mat_conf = data_processing.position_of_1_in_sublists(Y_test)
    cm = confusion_matrix(Y_pred, Y_test_mat_conf)

    acc = 0
    for i in range(len(cm)):
        acc += cm[i][i] / (cm[0][i] + cm[1][i] + cm[2][i])

    print("confusion matrix for test set {}".format(name))
    print(cm)
    print("final accuracy on test set")

    print(acc / len(cm))
    print("-----------------------------------------")
    print("-----------------------------------------")
    return acc / len(cm)


def adujsted_loss(Y_true, Y_pred):
    cm = confusion_matrix(Y_pred, Y_true)
    acc = 0
    for i in range(len(cm)):
        acc += cm[i][i] / (cm[0][i] + cm[1][i] + cm[2][i])
    return acc