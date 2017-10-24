"""Functions to show multiple results."""
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from data_processing import position_of_1_in_sublists
import accuracies_metrics


def infos(Y_test, Y_train):
    """Show infos about the set composition."""
    Y_test_set = position_of_1_in_sublists(Y_test)
    Y_train_set = position_of_1_in_sublists(Y_train)
    n_cats = len(Y_test[0])
    percentage_cats_test = {i : Y_test_set.count(i) / len(Y_test) for i in range(n_cats)}
    percentage_cats_train = {i : Y_train_set.count(i) / len(Y_train) for i in range(n_cats)}

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


def show_confusion_matrix_of_voting_methods(Y_test_pred_assemble, Y_test,
                                            groups_frames_names_test,
                                            Y_test_pred_vote,
                                            Y_test_pred_average_proba,
                                            Y_test_pred_boosting_MLP,
                                            Y_test_pred_MLP_on_classes,
                                            n_models):
    """Show different metrics about voting methods performance."""
    mlp_score = accuracies_metrics.percentage_accuracy_videos(
        Y_test_pred_assemble, Y_test,
        groups_frames_names=groups_frames_names_test)
    print("scores for MLP model: {0}".format(mlp_score))

    print("mat conf MLP model")
    a = accuracies_metrics.mat_conf(Y_test, Y_test_pred_assemble)
    print(a)

    print("mat conf of MLP, learning on classes predicted")
    e = accuracies_metrics.mat_conf(Y_test, Y_test_pred_MLP_on_classes)
    print(e)

    print("------------------------------------------------------------------")
    print("mat conf vote model")
    b = accuracies_metrics.mat_conf(Y_test, Y_test_pred_vote)
    print(b)

    print("------------------------------------------------------------------")
    print("mat conf average prob model")
    c = accuracies_metrics.mat_conf(Y_test, Y_test_pred_average_proba)
    print(c)

    print("------------------------------------------------------------------")

    d = 0  # Value by default
    if n_models != 1:
        print("--------------------------------------------------------------")
        print("mat conf boosting model")
        d = accuracies_metrics.mat_conf(Y_test, Y_test_pred_boosting_MLP)
        print(d)
