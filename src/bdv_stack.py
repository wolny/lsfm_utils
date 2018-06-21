import re
import sys

import h5py

"""
Usage:
python stack_bdv.py <Channel> <BigDataViewerFile> <OutputFile>
where channel in {0, 1}
"""
if len(sys.argv) != 4:
    print(
        "Usage: python stack_bdv.py <Channel> <BigDataViewerFile> <OutputFile>")
    sys.exit(1)

channel = f"s0{sys.argv[1]}"
bdv_input_file_path = sys.argv[2]
output_file = sys.argv[3]

f_input = h5py.File(bdv_input_file_path, "r")

cell_paths = []

for t in f_input.keys():
    if re.match("t[0-9]+", t):
        cell_paths.append(f"{t}/{channel}/0/cells")

if not cell_paths:
    print("No time points found in the BigDataViewer H5 file.")
    sys.exit(1)
else:
    print(f"Found {len(cell_paths)} time points: {cell_paths}")

first_cell = f_input[cell_paths[0]]
shape = (len(cell_paths),) + first_cell.shape
dtype = first_cell.dtype

f_output = h5py.File(output_file, "w")
dset = f_output.create_dataset(f"channel_{channel}",
                               shape=shape,
                               dtype=dtype,
                               chunks=True,
                               compression="gzip",
                               compression_opts=1
                               )

for t, path in enumerate(cell_paths):
    cell = f_input[path]
    # the axis order for a given time point is 'zyx', we want it to be 'xyz'
    print(f"Processing path '{path}'...")
    dset[t] = cell
