"""
Usage:
python stack_raw_and_ground_truth.py <raw_h5_file> <labels_h5_file> <output_h5_file>

Takes the '/channel_s00' dataset from the 'raw_h5_file', '/volume/labels/merged_ids'
from the 'labels_h5_file', and creates the output file 'output_h5_file' with
the following datasets:
1. '/volumes/raw' - raw data
2. '/volumes/labels' - labels
3. '/volumes/stacked' - raw data and labels stacked together along new axis
"""

import sys

import h5py
import numpy

if len(sys.argv) < 4:
    print(
        "Usage: python stack_raw_and_ground_truth.py <raw_h5_file> <labels_h5_file> <output_h5_file>"
    )
    sys.exit(1)


def parse_args(args):
    raw_h5_file = args[1]
    labels_h5_file = args[2]
    output_h5_file = args[3]
    return raw_h5_file, labels_h5_file, output_h5_file


raw_h5_file, labels_h5_file, output_h5_file = parse_args(sys.argv)
print(f"Raw data file: {raw_h5_file}")
print(f"Labels file: {labels_h5_file}")
print(f"Output file: {output_h5_file}")

with h5py.File(raw_h5_file, "r") as raw_h5:
    with h5py.File(labels_h5_file, "r") as labels_h5:
        with h5py.File(output_h5_file, "w") as output_h5:
            raw = raw_h5["/channel_s00"]
            labels = labels_h5["/volumes/labels/merged_ids"][...].astype(numpy.uint16)

            output_h5.create_dataset(
                '/volumes/stacked',
                data=numpy.stack([raw, labels], axis=-1),
                dtype=numpy.uint16,
                compression="gzip")
