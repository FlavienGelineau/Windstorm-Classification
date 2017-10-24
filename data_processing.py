"""General functions to process data."""
from PIL import Image
import numpy as np

from os import listdir
from os.path import isfile, join


def get_all_names_from_path(path):
    """Return the names of all the files from a given path.

    Inputs : path, string
    Outputs : a list of strings, which are the names of all files in the path
    """
    return [f for f in listdir(path) if isfile(join(path, f))]


def get_path(folder, name_file):
	"""Get path to import files."""
	with open('info_files/paths.txt') as f:
		content = f.readlines()
	name_to_path = {}
	for line in content:
		name, path = line.split()
		name_to_path[name] = path

	return name_to_path[folder] + name_file


def jpeg_to_array(infilename, size):
    """Return he array version of a jpeg picture."""
    img = Image.open(infilename)
    img = img.resize(size, Image.ANTIALIAS)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def position_of_1_in_sublists(Y):
    """Return the position of the 1 in each sublist."""
    return [Y[i].tolist().index(1) for i in range(len(Y))]


def max_in_each_list(Y):
    """Return the position of the max element in each sublist."""
    return [Y[i].tolist().index(max(Y[i])) for i in range(len(Y))]
