"""
Usage:
python complete_segments_uint16_converter.py <input_h5_file> <output_h5_file> [Dtype]

Takes the '/complete_segments' dataset from the 'input_h5_file', converts
it to the desired 'Dtype' and writes the converted dataset into the '/complete_segments'
in the 'output_h5_file'

Optional Dtype param specifies desired output type (by default 'uint16').
"""

import sys

import h5py
import numpy

if len(sys.argv) < 3:
    print(
        "Usage: python complete_segments_uint16_converter.py <input_h5_file> <output_h5_file> [Dtype]"
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

with h5py.File(input_h5_file, "r") as input_h5:
    with h5py.File(output_h5_file, "w") as output_h5:
        complete_segments = input_h5["/complete_segments"]
        output_h5.create_dataset(
            "/complete_segments",
            data=complete_segments,
            dtype=output_dtype,
            compression="gzip")
