"""Functions to show multiple results."""
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from data_processing import position_of_1_in_sublists
import accuracies_metrics
import numpy as np

def infos(Y_test, Y_train):
    """Show infos about the set composition."""
    Y_test_set = position_of_1_in_sublists(Y_test)
    Y_train_set = position_of_1_in_sublists(Y_train)
    n_cats = len(Y_test[0])
    percentage_cats_test = {i: Y_test_set.count(i) / len(Y_test) for i in range(n_cats)}
    percentage_cats_train = {i: Y_train_set.count(i) / len(Y_train) for i in range(n_cats)}

    print("-------------------------------------------------------------")
    print("percentage of windstorms among the train set : {0}".format(
        percentage_cats_train))
    print("percentage of windstorms among the test set  : {0}".format(
        percentage_cats_test))
    print("-------------------------------------------------------------")


def show_confusion_matrix(Y_test, Y_test_pred, Y_train, Y_train_pred):
    """Show confusion matrix with the data given."""
    Y_test_formated = position_of_1_in_sublists(Y_test)
    Y_train_formated = position_of_1_in_sublists(Y_train)

    cm = confusion_matrix(Y_test_formated, Y_test_pred)
    print("confusion matrix for test set")
    print(cm)
    print("-----------------------------------------")

    cm = confusion_matrix(Y_train_formated, Y_train_pred)
    print("confusion matrix for train set")
    print(cm)


def show_accuracy_over_time(history, title):
    """Show the evolution of accuracy over the epochs."""
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('accuracy of ' + str(title))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss of' + str(title))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def get_accs(Y_test, Y_test_pred_assemble, Y_test_pred_vote, Y_test_pred_average_proba, n_models,
             Y_test_pred_boosting_MLP, Y_test_pred_MLP_on_classes, Y_test_pred_boosting_MLP_on_classes):
    """Get accs of different voting methods."""
    acc_vote_prob = accuracies_metrics.mat_conf(Y_test, Y_test_pred_assemble, "vote with probas predicted")
    acc_vote_class = accuracies_metrics.mat_conf(Y_test, Y_test_pred_vote, "vote with class predicted")
    acc_MLP = accuracies_metrics.mat_conf(Y_test, Y_test_pred_average_proba, " MLP on probas predicted")
    acc_MLP_class = accuracies_metrics.mat_conf(Y_test, Y_test_pred_MLP_on_classes, "MLP on class, predicted")
    acc_boosting = 0
    acc_boosting_MLP_class = 0
    if n_models != 1:
        acc_boosting = accuracies_metrics.mat_conf(Y_test, np.array(Y_test_pred_boosting_MLP),
                                                   "boosting on probas predicted")
        acc_boosting_MLP_class = accuracies_metrics.mat_conf(
            Y_test, np.array(Y_test_pred_boosting_MLP_on_classes), "boosting on class predicted")

    return acc_vote_prob, acc_vote_class, acc_MLP, acc_boosting, acc_MLP_class, acc_boosting_MLP_class
