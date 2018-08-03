"""
Usage:
python small_region_filter.py <input_h5_file> <output_h5_file> <min_size>

Takes the '/volumes/labels/merged_ids' dataset from the 'input_h5_file' and
filter (assign label '0') all of the regions which are smaller than the 'min_size' pixels.
"""

import sys

import h5py
import numpy

import time

if len(sys.argv) < 4:
    print(
        "Usage: python small_region_filter.py <input_h5_file> <output_h5_file> <min_size>"
    )
    sys.exit(1)


def parse_args(args):
    input_h5_file = args[1]
    output_h5_file = args[2]
    min_size = int(args[3])
    return input_h5_file, output_h5_file, min_size


input_h5_file, output_h5_file, min_size = parse_args(sys.argv)
print(f"Input file: {input_h5_file}")
print(f"Output file: {output_h5_file}")
print(f"Min size: {min_size}")


def filter_label_ids(label_ids, min_size):
    label_ids_copy = numpy.copy(label_ids)

    exclusion_label = 0  # assign '0' to small regions

    unique_ids, counts = numpy.unique(label_ids_copy, return_counts=True)

    if unique_ids[0] == exclusion_label:
        print(
            "Warning: your exclusion label corresponds to one of the labels in your stack")

    # get labels for regions smaller than 'min_size' pixels
    small_region_ids = unique_ids[counts < min_size]

    print(
        f"Number of regions smaller than {min_size} pixels: {len(small_region_ids)}"
    )

    # map all small_region_ids to 'exclusion_label'
    max_label_id = unique_ids[-1]
    new_unique_ids = numpy.arange(0, max_label_id + 1)
    new_unique_ids[small_region_ids] = exclusion_label

    return new_unique_ids[label_ids_copy]


with h5py.File(input_h5_file, "r") as input_h5:
    with h5py.File(output_h5_file, "w") as output_h5:
        label_ids = input_h5["/volumes/labels/merged_ids"]

        start = time.time()

        filtered_label_ids = filter_label_ids(label_ids, min_size)

        end = time.time()

        print("Small region filtering took %.3f" % (end - start))

        output_h5.create_dataset(
            "/volumes/labels/merged_ids",
            data=filtered_label_ids,
            dtype=numpy.uint16,
            compression="gzip")
