"""Test of import_set functions."""

import import_set


def test_make_Y():
    """Test make_Y."""
    labels = [1, 2]
    n_groups_of_frames_per_video = [4, 5]
    obtained = import_set.make_Y(labels, n_groups_of_frames_per_video)

    expected = [1, 1, 1, 1, 2, 2, 2, 2, 2]
    assert obtained == expected


def test_equalize_proportions_labels():
    names_train = '209-0.mp4', '165-2.mp4', '283-2.mp4', '56-0.mp4', '226-2.mp4', '239-2.mp4', '259-1.mp4', '249-1.mp4'

    expected = (['209-0.mp4', '165-2.mp4', '283-2.mp4', '56-0.mp4', '259-1.mp4', '249-1.mp4'],
                ['226-2.mp4', '239-2.mp4'])
    obtained = import_set.equalize_proportions_labels(names_train)
    assert obtained == expected

test_equalize_proportions_labels()
