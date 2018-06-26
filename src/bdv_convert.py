import h5py
import re
import sys
import numpy as np
import os

"""
Usage:
python bdv_convert.py <Channel> <BigDataViewerFile> [Dtype] 
where channel in {0, 1}. Optional Dtype param specifies desired output type (e.g. uint8).

Output: one h5 file per time-point
"""
if len(sys.argv) < 3:
    print(
        "Usage: python bdv_convert.py <Channel> <BigDataViewerFile> [Dtype]")
    sys.exit(1)


def parse_args(args):
    channel = f"s0{args[1]}"
    bdv_input_file_path = args[2]
    output_dtype = None
    if len(args) > 3:
        output_dtype = np.dtype(args[3])
    return channel, bdv_input_file_path, output_dtype


def get_cell_paths(h5file, channel):
    cell_paths = []
    for t in h5file.keys():
        if re.match("t[0-9]+", t):
            cell_paths.append(f"{t}/{channel}/0/cells")
    return cell_paths


def file_name_from_path(path, dtype):
    tmp = path.split(os.path.sep)[:-2]
    tmp.append(str(np.dtype(dtype)))
    return '_'.join(tmp) + '.h5'


def convert(cell, dtype):
    if dtype != np.uint8:
        raise RuntimeError("Only conversion to 'uint8' is supported")

    min_intensity = np.amin(cell)
    max_intensity = np.amax(cell)
    cell -= min_intensity
    return cell // ((max_intensity - min_intensity + 1) / 256)


channel, bdv_input_file_path, output_dtype = parse_args(sys.argv)

f_input = h5py.File(bdv_input_file_path, "r")

cell_paths = get_cell_paths(f_input, channel)

if not cell_paths:
    print("No time points found in the BigDataViewer H5 file.")
    sys.exit(1)
else:
    print(f"Found {len(cell_paths)} time points: {cell_paths}")

first_cell = f_input[cell_paths[0]]
shape = first_cell.shape

if output_dtype is not None:
    dtype = output_dtype
else:
    dtype = first_cell.dtype

for path in cell_paths:
    print(f"Processing path '{path}'...")
    output_file = file_name_from_path(path, dtype)
    with h5py.File(output_file, "w") as f_output:
        dset = f_output.create_dataset(f"channel_{channel}",
                                       shape=shape,
                                       dtype=dtype,
                                       chunks=True,
                                       compression="gzip",
                                       compression_opts=1)
        cell = f_input[path]
        dset[:] = convert(cell, dtype)
