import sys

import h5py
import numpy

if len(sys.argv) < 2:
    print(
        "Usage: python squeeze_dims.py <input_h5_file>"
    )
    sys.exit(1)

input_h5_file = sys.argv[1]

with h5py.File(input_h5_file, "r+") as input_h5:
    gt = numpy.squeeze(input_h5["data"])

    datasets = {
        '/volumes/raw': gt[..., 0],
        '/volumes/labels': gt[..., 1],
        '/volumes/stacked': gt
    }

    for k, v in datasets.items():
        input_h5.create_dataset(
            k,
            data=v,
            dtype=gt.dtype,
            compression="gzip")

    del input_h5['data']
