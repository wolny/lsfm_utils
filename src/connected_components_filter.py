"""
Usage:
python connected_components_filter.py <input_h5_file> <output_h5_file>

Due to issues with bigcat some of the regions (region - group of pixels with a particular label id)
are disconnected. We need to find those regions and keep only the biggest part of such a region.

The script takes the '/volumes/labels/merged_ids' dataset from the 'input_h5_file',
for each label it looks for the connected components, if the number is bigger than 1,
filter (assign label '0') except the biggest one.
"""

import sys

import h5py
import numpy

import time
from scipy.ndimage import label

if len(sys.argv) < 3:
    print(
        "Usage: python connected_components_filter.py <input_h5_file> <output_h5_file>"
    )
    sys.exit(1)


def parse_args(args):
    input_h5_file = args[1]
    output_h5_file = args[2]
    return input_h5_file, output_h5_file


input_h5_file, output_h5_file, = parse_args(sys.argv)
print(f"Input file: {input_h5_file}")
print(f"Output file: {output_h5_file}")


def filter_connected_components(label_ids):
    label_ids_copy = numpy.copy(label_ids)

    unique_ids = numpy.unique(label_ids_copy)

    filter_mask = numpy.ones(label_ids_copy.shape)

    # skip 0-label
    for label_id in unique_ids[1:]:
        mask = (label_ids_copy == label_id) * 1
        components, num_components = label(mask)
        if num_components > 1:
            print(
                f"Found {num_components} disconnected regions with label {label_id}. Filtering...")
            u_ids, counts = numpy.unique(components,
                                         return_counts=True)
            # delete 0-label
            ind = numpy.where(u_ids == 0)[0][0]
            u_ids = numpy.delete(u_ids, [ind])
            counts = numpy.delete(counts, [ind])

            # delete biggest region
            max_label = u_ids[numpy.argmax(counts)]
            ind = numpy.where(u_ids == max_label)[0][0]
            u_ids = numpy.delete(u_ids, [ind])
            counts = numpy.delete(counts, [ind])

            # filter remanding labels
            for l_id in u_ids:
                filter_mask[components == l_id] = 0

    # zero out small connected components
    return label_ids_copy * filter_mask


with h5py.File(input_h5_file, "r") as input_h5:
    with h5py.File(output_h5_file, "w") as output_h5:
        label_ids = input_h5["/volumes/labels/merged_ids"]

        start = time.time()

        filtered_label_ids = filter_connected_components(label_ids)

        end = time.time()

        print("Small region filtering took %.3f" % (end - start))

        output_h5.create_dataset(
            "/volumes/labels/merged_ids",
            data=filtered_label_ids,
            dtype=numpy.uint16,
            compression="gzip")
