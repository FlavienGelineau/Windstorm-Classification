import import_set

def test_make_Y():
	labels = [1, 2]
	n_groups_of_frames_per_video = [4, 5]
	obtained = import_set.make_Y(labels, n_groups_of_frames_per_video)

	expected = [1,1,1,1,2,2,2,2,2]
	assert obtained == expected