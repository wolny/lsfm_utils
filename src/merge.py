import os
import sys
import numpy
import h5py


def get_raw_and_segmented_time_points(raw_data_dir, segmented_data_dir):
    raw_dir = os.fsencode(raw_data_dir)
    raw_time_points = []
    segmented_time_points = []

    for file in os.listdir(raw_dir):
        filename = os.fsdecode(file)
        if filename.endswith("uint8.h5"):
            raw_time_points.append(os.path.join(raw_data_dir, filename))
            segmented_filename = os.path.splitext(filename)[
                                     0] + '_Multicut Segmentation.h5'
            segmented_time_points.append(
                os.path.join(segmented_data_dir, segmented_filename))

    return raw_time_points, segmented_time_points


def merge_raw_with_segmented(raw_data_path, segmented_data_path, output_dir):
    with h5py.File(raw_data_path, "r") as raw_data:
        with h5py.File(segmented_data_path, "r") as segmented_data:
            raw_filename = os.path.split(raw_data)[1]
            raw_filename = os.path.splitext(raw_filename)[0]
            with h5py.File(
                    os.path.join(output_dir, raw_filename + '_raw_and_seg.h5'),
                    'w') as output_h5:
                raw_dataset = raw_data['channel_s00']
                segmented_dataset = segmented_data['exported_data']
                raw_and_segmented = numpy.stack(
                    [raw_dataset, segmented_dataset])
                output_h5.create_dataset(
                    "raw_and_segmented",
                    data=raw_and_segmented,
                    dtype=numpy.uint16,
                    compression="gzip")


if len(sys.argv) != 4:
    print("Usage: python merge.py <RawDataDir> <SegmentationDir> <OutputDir>")
    sys.exit(1)

raw_data_dir = sys.argv[1]
segmented_data_dir = sys.argv[2]
output_dir = sys.argv[3]

raw_time_points, segmented_time_points = get_raw_and_segmented_time_points(
    raw_data_dir, segmented_data_dir)

print(f'Raw data: {raw_time_points}')
print(f'Segmented data: {segmented_time_points}')

if not (raw_time_points or segmented_time_points):
    print(f"No raw data or segmentation found at {raw_data_dir}")
    sys.exit(1)

for raw_data_path, segmented_data_path in zip(raw_time_points,
                                              segmented_time_points):
    if not segmented_data_path:
        print(f"WARNING: Cannot find multicut segmentation for {raw_data_path}")
    else:
        print(
            f"Merging raw data '{raw_data_path}' with segmentation '{segmented_data_path}'")
        merge_raw_with_segmented(raw_data_path, segmented_data_path, output_dir)
