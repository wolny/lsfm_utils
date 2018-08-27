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
        if filename.endswith("uint8.h5"):
            raw_time_points.add(os.path.join(dir_str, filename))
        elif filename.endswith("Multicut Segmentation.h5"):
            segmented_time_points.add((os.path.join(dir_str, filename)))

    return raw_time_points, segmented_time_points


def get_segmented_file(raw_data_path, segmented_time_points):
    path = raw_data_path.split('.')[0] + "_Multicut Segmentation.h5"
    if path in segmented_time_points:
        return path
    else:
        return None


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

raw_time_points, segmented_time_points = get_raw_and_segmented_time_points(
    dir_str)

if not (raw_time_points or segmented_time_points):
    print(f"No raw data or segmentation found at {dir_str}")
    sys.exit(1)

for raw_data_path in raw_time_points:
    segmented_file_path = get_segmented_file(raw_data_path,
                                             segmented_time_points)
    if not segmented_file_path:
        print(f"WARNING: Cannot find multicut segmentation for {raw_data_path}")
    else:
        print(
            f"Merging raw data '{raw_data_path}' with '{segmented_file_path}'")
        merge_raw_with_segmented(raw_data_path, segmented_file_path)
