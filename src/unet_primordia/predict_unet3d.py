#!/g/kreshuk/wolny/miniconda3/envs/pytorch041/bin/python3

import argparse
import os

import h5py
import numpy as np
import torch
from primordia_loader.loader import get_raw_volumes
from unet3d import utils
from unet3d.model import UNet3D

logger = utils.get_logger('UNet3DPredictor')


def predict(model, dataset, device):
    """
    Return prediction masks by applying the model on the given dataset

    Args:
        model (Unet3D): trained UNet 3D model used for prediction
        dataset (torch.utils.data.Dataset): input dataset
        device (torch.Device): device to run the prediction on

    Returns:
         probability_maps: numpy array of the same shape as the 'dataset'
    """
    logger.info(
        f'Running prediction on dataset {dataset.name}: {dataset.path}...')
    # number of input channels expected by the model
    in_channels = model.in_channels
    # number of output channels expected by the model
    out_channels = model.out_channels
    # dimensionality of the the output
    probability_maps_shape = (out_channels,) + dataset.volume.shape
    # initialize the output prediction array
    probability_maps = np.zeros(probability_maps_shape, dtype='float32')

    # initialize normalization mask in order to average out probabilities
    # of overlapping patches
    normalization_mask = np.zeros(probability_maps_shape, dtype='float32')

    with torch.no_grad():
        for patch, index_spec in dataset:
            logger.info(f'Predicting slice:{index_spec.base_sequence_at_index}')

            # save patch index: (C,) + (D,H,W)
            channel_slice = slice(0, out_channels)
            index = (channel_slice,) + index_spec.base_sequence_at_index

            # convert patch to torch tensor NxCxDxHxW and send to device
            patch = torch \
                .from_numpy(patch) \
                .view((1, in_channels) + patch.shape) \
                .to(device)

            # forward pass
            probs = model(patch)
            # convert back to numpy array
            probs = probs.squeeze().cpu().numpy()
            # accumulate probabilities into the output prediction array
            probability_maps[index] += probs
            # count voxel visits for normalization
            normalization_mask[index] += 1

    return probability_maps / normalization_mask


def save_predictions(probability_maps, output_file, average_all_channels=True):
    """
    Saving probability maps to a given output H5 file. If 'average_all_channels'
    is set to True average the probability_maps across the the channel axis,
    otherwise average out 3 consecutive channels from probability_maps
    (this is because the output channels from the model predict edge affinities
    in different axis and different offsets, e.g. for two offsets 'd1' and 'd2'
    we would have 6 channels corresponding to different axis/offset combination:
    x_d1, y_d1, z_d1, x_d2, y_d2, z_d2. When averaged it would produce output
    channels one for 'd1' and one for 'd2') and write them to separate datasets
    in the output H5 file.

    Args:
        probability_maps (numpy.ndarray): numpy array containing probability
            maps in separate channels
        output_file (string): path to the output H5 file
        average_all_channels (bool): if True average of the channels in the
            probability_maps otherwise average 3 consecutive channels from
            probability_maps
    """
    logger.info(f'Saving predictions to: {output_file}')

    def _dataset_dict():
        result = {}
        if average_all_channels:
            result['probability_maps'] = np.mean(probability_maps, axis=0)
        else:
            out_channels = probability_maps.shape[0]
            # iterate 3 channels at a time
            for i, c in enumerate(range(0, out_channels, 3)):
                # average 3 consecutive channels
                avg_probs = np.mean(probability_maps[c:c + 3, ...], axis=0)
                result[f'probability_maps{i}'] = avg_probs
        return result

    with h5py.File(output_file, "w") as output_h5:
        for k, v in _dataset_dict().items():
            logger.info(f'Creating dataset {k}')
            output_h5.create_dataset(k, data=v, dtype=v.dtype,
                                     compression="gzip")


def main():
    def _get_output_file(dataset):
        volume_file, volume_ext = os.path.splitext(dataset.path)
        return volume_file + '_probabilities' + volume_ext

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, help='path to the model')
    parser.add_argument('--config-path', type=str,
                        help='path to the dataset config')
    parser.add_argument('--in-channels', default=1, type=int,
                        help='number of input channels')
    parser.add_argument('--out-channels', default=6, type=int,
                        help='number of output channels')
    parser.add_argument('--layer-order', type=str,
                        help="Conv layer ordering, e.g. 'brc' -> BatchNorm3d+ReLU+Conv3D",
                        default='brc')
    parser.add_argument('--interpolate',
                        help='use F.interpolate instead of ConvTranspose3d',
                        action='store_true')

    args = parser.parse_args()

    # make sure those values correspond to the ones used during training
    in_channels = args.in_channels
    out_channels = args.out_channels
    # use F.interpolate for upsampling
    interpolate = args.interpolate
    # Conv layer ordering e.g. 'cr' is equivalent to Conv3D+ReLU
    conv_layer_order = args.layer_order
    model = UNet3D(in_channels, out_channels, interpolate=interpolate,
                   final_sigmoid=True, conv_layer_order=conv_layer_order)

    logger.info(f'Loading model from {args.model_path}...')
    utils.load_checkpoint(args.model_path, model)

    logger.info('Loading datasets...')
    raw_volumes = get_raw_volumes(args.config_path)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        logger.warning(
            'No CUDA device available. Using CPU for predictions')
        device = torch.device('cpu')

    model = model.to(device)

    for raw_volume in raw_volumes:
        probability_maps = predict(model, raw_volume, device)

        output_file = _get_output_file(raw_volume)

        save_predictions(probability_maps, output_file)


if __name__ == '__main__':
    main()
