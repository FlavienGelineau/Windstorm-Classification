"""Test vote methods."""

import voting_methods
import numpy as np


def test_boosting_MLP_class_predicted():
    """Test boosting_MLP_class_predicted."""
    X_train = np.array([
        [0.1, 0.2, 0.7, 0.4, 0.45, 0.2, 0.4, 0.45, 0.2],
        [0.1, 0.2, 0.7, 0.4, 0.45, 0.2, 0.4, 0.45, 0.2],
    ])
    n_classes = 3
    n_models_together = 2
    obtained = voting_methods.get_X_boosting_classes(
        X_train, n_classes, n_models_together=n_models_together)
    expected = [
        [2, 1],
        [2, 1],
    ]
    assert obtained == expected, print(obtained, expected)


test_boosting_MLP_class_predicted()
