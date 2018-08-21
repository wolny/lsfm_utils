import argparse
import logging
import os
import sys

import h5py
import numpy as np
import torch
from primordia_loader.loader import get_test_dataset

from inferno.trainers.basic import Trainer

logging.basicConfig(format='[%(asctime)-15s][%(name)s %(levelname)s]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def load_model(model_dir):
    logger.info(f'Loading model from: {model_path}')
    model = Trainer().load_model(model_dir, 'best_checkpoint.pytorch').model
    return model


def predict(model, dataset):
    # initialize the output prediction array
    output = np.zeros((model.out_channels,) + dataset.volume.shape,
                      dtype='float32')
    device = torch.device('cuda:0')

    with torch.no_grad():
        for patch, index_spec in dataset:
            logger.info(f'Predicting slice:{index_spec.base_sequence_at_index}')
            # (C,) + (D,H,W)
            index = (slice(0,
                           model.out_channels),) + index_spec.base_sequence_at_index
            # convert to torch tensor 1xCxDxHxW
            t_shape = (1,) + (model.in_channels,) + patch.shape
            patch = torch.from_numpy(patch).view(t_shape).to(device)
            # forward pass
            probs = model(patch)
            # convert to numpy array
            probs = probs.squeeze().cpu().numpy()
            # write to the output prediction array
            output[index] = probs

    return output


def main():
    test_config = './config/test_config.yml'
    train_config = './config/train_config.yml'

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str)
    parser.add_argument('--output_file', type=str)

    args = parser.parse_args()

    project_dir = args.project_dir
    output_file = args.output_file

    model_dir = os.path.join(project_dir, 'Weights')

    logger.info('Loading dataset...')
    raw_volume = get_test_dataset(test_config)
    model = load_model(model_dir)

    output = predict(model, raw_volume)

    save_predictions(output, output_file)


def save_predictions(output, output_file):
    logger.info(f'Saving predictions to: {output_file}')
    with h5py.File(output_file, "w") as output_h5:
        output_h5.create_dataset(
            'probability_maps',
            data=output,
            dtype=output.dtype,
            compression="gzip")


if __name__ == '__main__':
    main()
