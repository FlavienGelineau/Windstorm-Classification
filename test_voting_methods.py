"""Test vote methods."""

import voting_methods
import numpy as np


def test_get_X_boosting_classes():
    """Test boosting_MLP_class_predicted."""
    X_train = np.array([
        [0.1, 0.2, 0.7, 0.4, 0.45, 0.2, 0.4, 0.45, 0.2],
        [0.1, 0.2, 0.7, 0.4, 0.45, 0.2, 0.4, 0.45, 0.2],
    ])
    n_classes = 3
    obtained = voting_methods.get_X_boosting_classes(X_train, n_classes, n_models = 3)
    expected = [
        [2, 1, 1],
        [2, 1, 1],
    ]
    assert obtained == expected


def test_boosting_MLP_class_predicted():
    X_train = np.array([
        [0.1, 0.2, 0.7, 0.4, 0.45, 0.2, 0.4, 0.45, 0.2],
        [0.1, 0.2, 0.7, 0.4, 0.45, 0.2, 0.4, 0.45, 0.2],
    ])
    X_test = np.array([
        [0.1, 0.2, 0.7, 0.4, 0.45, 0.2, 0.4, 0.45, 0.2],
        [0.1, 0.2, 0.7, 0.4, 0.45, 0.2, 0.4, 0.45, 0.2],
    ])
    Y_train, Y_test = [[0,0,1],[0,0,1]], [[0,0,1],[0,0,1]]
    n_models = 3
    n_classes = 3
    n_models_together = 2

    voting_methods.boosting_MLP_class_predicted(X_train, X_test, Y_train, Y_test, n_models, n_classes, n_models_together)

test_get_X_boosting_classes()
test_boosting_MLP_class_predicted()
