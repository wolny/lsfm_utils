import argparse
import logging
import os
import sys

import h5py
import numpy as np
import torch
from primordia_loader.loader import get_test_dataset

import neurofire.models as models
from inferno.utils.io_utils import yaml2dict

logging.basicConfig(format='[%(asctime)-15s][%(name)s %(levelname)s]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def load_model(train_config, model_path):
    model_config = yaml2dict(train_config)

    model_name = model_config.get('model_name')
    model = getattr(models, model_name)(**model_config.get('model_kwargs'))
    model.cuda()
    model.load_state_dict(torch.load(model_path))
    return model


def predict(model, dataset):
    output = np.zeros((model.out_channels,) + dataset.volume.shape,
                      dtype='float32')
    device = torch.device('cuda:0')

    with torch.no_grad():
        for patch, index_spec in dataset:
            logger.info(f'Predicting slice:{index_spec.base_sequence_at_index}')
            # (C,) + (D,H,W)
            index = (slice(0,
                           model.out_channels),) + index_spec.base_sequence_at_index
            # convert to torch tensor
            t_shape = (1,) + (model.in_channels,) + patch.shape
            patch = torch.from_numpy(patch).view(t_shape).to(device)
            # forward pass
            probs = model(patch)
            # convert to numpy array
            probs = probs.squeeze().cpu().numpy()
            # write to output array
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

    model_path = os.path.join(project_dir, 'Weights/best_checkpoint.pytorch')

    raw_volume = get_test_dataset(test_config)
    model = load_model(train_config, model_path)

    output = predict(model, raw_volume)

    with h5py.File(output_file, "w") as output_h5:
        output_h5.create_dataset(
            'probability_maps',
            data=output,
            dtype=output.dtype,
            compression="gzip")


if __name__ == '__main__':
    main()
