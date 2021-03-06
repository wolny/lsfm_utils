"""
Usage:
python label_ids_converter.py <input_h5_file> <output_h5_file> [Dtype]

Takes the '/volumes/labels/merged_ids' dataset from the 'input_h5_file', squeezes
the label ids (see below), converts the new labels to the desired 'Dtype'
and writes the converted dataset into the '/volumes/labels/merged_ids'
in the 'output_h5_file'.

Optional Dtype param specifies desired output type (by default 'uint16').
"""

import sys

import h5py
import numpy

import time

if len(sys.argv) < 3:
    print(
        "Usage: python label_ids_converter.py <input_h5_file> <output_h5_file> [Dtype]"
    )
    sys.exit(1)


def parse_args(args):
    input_h5_file = args[1]
    output_h5_file = args[2]
    output_dtype = numpy.uint16
    if len(args) > 3:
        output_dtype = numpy.dtype(args[3])
    return input_h5_file, output_h5_file, output_dtype


input_h5_file, output_h5_file, output_dtype = parse_args(sys.argv)


def squeeze_label_ids(label_ids):
    """
    Takes the 'label_ids' numpy array converts it to a set in order to obtain unique values,
    then creates a mapping from old label_ids to a consecutive integers (starting from 1)
    and in the end returns the numpy array created from the original 'label_ids' with
    the values remapped.
    E.g. array [7, 7, 2, 3, 2, 5] will return [1, 1, 2, 3, 2, 4], i.e. the original labels are squeezed.
    :param label_ids: input numpy array
    :return: numpy array of the same shape as label_ids with the original values remapped
    """
    label_ids_copy = numpy.copy(label_ids)
    unique_ids, indices = numpy.unique(label_ids_copy, return_inverse=True)
    num_label_ids = len(unique_ids)
    print(f"Number of unique label ids: {num_label_ids}")
    # just use the range [1:num_label_ids+1] as new label ids
    unique_ids = numpy.arange(1, num_label_ids + 1)
    # reconstruct the label_ids array with the new labels
    return unique_ids[indices].reshape(label_ids_copy.shape)


with h5py.File(input_h5_file, "r") as input_h5:
    with h5py.File(output_h5_file, "w") as output_h5:
        label_ids = input_h5["/volumes/labels/merged_ids"]

        start = time.time()

        squeezed_label_ids = squeeze_label_ids(label_ids)

        end = time.time()

        print("Label squeezing took %.3f" % (end - start))

        max_label_id = numpy.max(squeezed_label_ids)
        if max_label_id > numpy.iinfo(output_dtype).max:
            print(
                f"Error: cannot convert label ids. Max label id '{max_label_id}' cannot be converted to {output_dtype}")
            sys.exit(1)

        output_h5.create_dataset(
            "/volumes/labels/merged_ids",
            data=squeezed_label_ids,
            dtype=output_dtype,
            compression="gzip")
