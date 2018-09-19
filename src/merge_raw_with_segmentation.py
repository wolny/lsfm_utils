"""
Usage:
python merge_raw_with_segmentation.py <DataDir>
For each time point H5 raw data it takes the corresponding segmentation output file
(files with suffix '_Multicut Segmentation.h5') and saves the segmentation
inside the raw H5 data file. Additionally creates the necessary Bigcat attributes,
so that in the end the files are ready for proofreading with Bigcat.

"""

import os
import sys
import numpy
import h5py


def get_raw_and_segmented_time_points(dir_str):
    dir = os.fsencode(dir_str)
    raw_time_points = set()
    segmented_time_points = set()

    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if filename.endswith("h5"):
            if filename.endswith("Multicut Segmentation.h5"):
                segmented_time_points.add((os.path.join(dir_str, filename)))
            else:
                raw_time_points.add(os.path.join(dir_str, filename))

    raw_to_segmented = {}
    for raw_file in raw_time_points:
        fn = os.path.split(raw_file)[1]
        fn_prefix = fn.split('_')[0]
        for segmented_file in segmented_time_points:
            fn = os.path.split(segmented_file)[1]
            if fn.startswith(fn_prefix):
                raw_to_segmented[raw_file] = segmented_file

    return raw_to_segmented


def merge_raw_with_segmented(raw_data_path, segmented_file_path):
    with h5py.File(raw_data_path, "r+") as raw_data:
        with h5py.File(segmented_file_path, "r") as segmented_file:
            # add necessary attributes to the raw dataset
            raw_data["channel_s00"].attrs["resolution"] = (0.25, 0.165, 0.165)

            # get segmented stack and reshape if necessary
            exported_data = numpy.squeeze(segmented_file["exported_data"])

            # create labels in the raw data file
            if "/volumes/labels/" not in raw_data:
                raw_data.create_group("/volumes/labels/")

            raw_data.create_dataset(
                "/volumes/labels/segment_ids",
                data=exported_data,
                dtype=numpy.uint64,
                compression="gzip")

            # add necessary attributes to the labels dataset
            raw_data["/volumes/labels/segment_ids"].attrs["resolution"] = \
                (0.25, 0.165, 0.165)
            raw_data["/volumes/labels/segment_ids"].attrs["offset"] = \
                (0.0, 0.0, 0.0)


if len(sys.argv) != 2:
    print("Usage: python merge_raw_with_segmentation.py <DataDir>")
    sys.exit(1)

dir_str = sys.argv[1]

raw_to_segmented = get_raw_and_segmented_time_points(dir_str)

if not raw_to_segmented:
    print(f"No raw data or segmentation found at {dir_str}")
    sys.exit(1)

for raw_data_path, segmented_file_path in raw_to_segmented.items():
    print(f"Merging raw data '{raw_data_path}' with '{segmented_file_path}'")
    merge_raw_with_segmented(raw_data_path, segmented_file_path)
