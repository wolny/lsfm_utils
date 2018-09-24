"""
Usage:
python add_offset_and_resolution.py <DataDir> <Dataset>

For each h5 file in the 'DataDir' add 'offset' and 'resolution' attributes
to the 'Dataset' inside the file.
"""

import os
import h5py
import sys


def add_offset_and_resolution(dir_str, dataset, offset=(0.0, 0.0, 0.0),
                              resolution=(0.25, 0.1625, 0.1625)):
    dir = os.fsencode(dir_str)

    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if filename.endswith("h5"):
            path = os.path.join(dir, filename)
            with h5py.File(path, "r+") as h5_file:
                h5_file[dataset].attrs["resolution"] = resolution
                h5_file[dataset].attrs["offset"] = offset


if len(sys.argv) != 3:
    print("Usage: python add_offset_and_resolution.py <DataDir> <Dataset>")
    sys.exit(1)

dir_str = sys.argv[1]
dataset = sys.argv[2]

add_offset_and_resolution(dir_str, dataset)
