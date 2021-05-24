from os import name
import h5py
import sys
import pdb
import re

file1 = sys.argv[1]
file2 = sys.argv[2]

# file1 = "h5data/Bindercrossings.h5"
# file2 = "h5data/Bindercrossings_N3.h5"


def copy_data(group1, group2):
    for key in group2.keys():
        if type(group2[key]) == h5py._hl.group.Group:

            if key not in group1.keys():
                group1.create_group(key)

            copy_data(group1[key], group2[key])

        elif type(group2[key]) == h5py._hl.dataset.Dataset:
            if key in group1.keys():
                del group1[key]

            group1.create_dataset(key, group2[key].shape, dtype=group2[key].dtype)

            group1[key][()] = group2[key][()]


with h5py.File(file1, "r+") as f1:
    with h5py.File(file2, "r") as f2:
        copy_data(f1, f2)
