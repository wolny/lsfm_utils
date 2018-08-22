import argparse
import logging
import os
import sys

import h5py
import numpy as np
import torch
from primordia_loader.loader import get_raw_volume

from inferno.trainers.basic import Trainer

logging.basicConfig(format='[%(asctime)-15s][%(name)s %(levelname)s]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def load_model(model_dir):
    logger.info(f'Loading best checkpoint from: {model_dir}...')
    model = Trainer().load(from_directory=model_dir, best=True,
                           filename='best_checkpoint.pytorch').model
    return model


def predict(model, dataset, device):
    logger.info(f'Running prediction on dataset {dataset.name}...')
    # number of input channels expected by the model
    in_channels = model.in_channels
    # number of output channels expected by the model
    out_channels = model.out_channels
    # initialize the output prediction array
    probability_maps = np.zeros((out_channels,) + dataset.volume.shape,
                                dtype='float32')
    # initialize normalization mask in order to average out probabilities
    # of overlapping patches
    normalization_mask = np.zeros((out_channels,) + dataset.volume.shape,
                                  dtype='float32')

    with torch.no_grad():
        for patch, index_spec in dataset:
            logger.info(f'Predicting slice:{index_spec.base_sequence_at_index}')
            # save patch index: (C,) + (D,H,W)
            index = (slice(0, out_channels),) + \
                    index_spec.base_sequence_at_index

            # convert patch to torch tensor NxCxDxHxW and send to device
            patch = torch \
                .from_numpy(patch) \
                .view((1, in_channels) + patch.shape) \
                .to(device)
            # forward pass
            probs = model(patch)
            # convert to numpy array
            probs = probs.squeeze().cpu().numpy()
            # write to the output prediction array
            probability_maps[index] = probs
            # count visits
            normalization_mask[index] += 1

    return probability_maps / normalization_mask


def save_predictions(probability_maps, output_file, average_channels=True):
    """
    Saving probability maps to a given H5 file. If 'aggregate_channels' is set
    to True also average out 3 consecutive channels (this is because the output
    channels from the model predict edge affinities in different axis
    and different offsets, e.g. for two offsets 'd1' and 'd2' we would have
    6 channels corresponding to different axis/offset combination:
    x_d1, y_d1, z_d1, x_d2, y_d2, z_d2
    """
    logger.info(f'Saving predictions to: {output_file}')

    def _dataset_dict(p_maps, a_channels):
        if not a_channels:
            return {'probability_maps': probability_maps}
        else:
            out_channels = p_maps.shape[0]
            result = {}
            for i, c in enumerate(range(0, out_channels, 3)):
                avg_probs = np.mean(p_maps[c:c + 3, ...], axis=0)
                result[f'probability_maps{i}'] = avg_probs
            return result

    with h5py.File(output_file, "w") as output_h5:
        for k, v in _dataset_dict(probability_maps, average_channels).items():
            logger.info(f'Creating dataset {k}')
            output_h5.create_dataset(k, data=v, dtype=v.dtype,
                                     compression="gzip")


def main():
    test_config = './config/test_config.yml'

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str)
    parser.add_argument('--output_file', type=str)

    args = parser.parse_args()

    project_dir = args.project_dir
    output_file = args.output_file

    model_dir = os.path.join(project_dir, 'Weights')

    logger.info('Loading RawVolume...')
    raw_volume = get_raw_volume(test_config)

    model = load_model(model_dir)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        logger.warning(
            'No CUDA device available. Predictions will be slooow...')
        device = torch.device('cpu')

    probability_maps = predict(model, raw_volume, device)

    save_predictions(probability_maps, output_file)


if __name__ == '__main__':
    main()
