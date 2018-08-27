import os
import sys

import h5py
import numpy


def convert_files(data_dir):
    encoded_data_dir = os.fsencode(data_dir)
    for file in os.listdir(encoded_data_dir):
        filename = os.fsdecode(file)
        segm_path = os.path.join(data_dir, filename)
        with h5py.File(segm_path, 'r+') as segm_h5:
            print('Converting ', segm_path)
            dataset = numpy.squeeze(segm_h5['exported_data'])
            del segm_h5['exported_data']
            segm_h5.create_dataset(
                'exported_data',
                data=dataset,
                dtype=numpy.uint16,
                compression='gzip'
            )


if len(sys.argv) != 2:
    print("Usage: python convert.py <SegmentationDir>")
    sys.exit(1)

segmented_data_dir = sys.argv[1]

convert_files(segmented_data_dir)
