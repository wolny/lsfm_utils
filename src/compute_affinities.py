import h5py
from affogato.affinities import compute_affinities
import numpy
import time
import sys

"""
Usage:
python compute_affinities.py <input_h5_file> <output_h5_file>

Takes the '/volumes/labels/merged_ids' dataset from the 'input_h5_file' 
computes the affinities in the specified directions and writes the results
into the specified 'output_h5_file'. Assumes that the input stack was already
filtered and the 'ignore label' is present and its value is 0.
"""

if len(sys.argv) < 3:
    print(
        "Usage: python compute_affinities.py <input_h5_file> <output_h5_file>"
    )
    sys.exit(1)


def parse_args(args):
    input_h5_file = args[1]
    output_h5_file = args[2]
    return input_h5_file, output_h5_file


input_h5_file, output_h5_file = parse_args(sys.argv)

print(f"Computing affinities based on {input_h5_file}")

with h5py.File(input_h5_file, "r+") as input_h5:
    label_ids = numpy.copy(input_h5["/volumes/labels/merged_ids"])

    start = time.time()
    affinities, mask = compute_affinities(label_ids,
                                          offset=[
                                              [-1, 0, 0],
                                              [0, -1, 0],
                                              [0, 0, -1],

                                              [-7, 0, 0],
                                              [0, -7, 0],
                                              [0, 0, -7],

                                              [-15, 0, 0],
                                              [0, -15, 0],
                                              [0, 0, -15]
                                          ],
                                          have_ignore_label=True,
                                          ignore_label=0)

    end = time.time()

    print("Computing affinities took %.3f" % (end - start))

    with h5py.File(output_h5_file, "w") as output_h5:
        output_h5.create_dataset("affinities", data=affinities,
                                 compression="gzip")
        output_h5.create_dataset("mask", data=mask, compression="gzip")
