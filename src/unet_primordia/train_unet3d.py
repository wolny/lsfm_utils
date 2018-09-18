#!/g/kreshuk/wolny/miniconda3/envs/pytorch041/bin/python3

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from primordia_loader.loader import get_primordia_loaders
from unet3d.model import UNet3D
from unet3d.trainer import UNet3DTrainer
from unet3d.utils import DiceCoefficient
from unet3d.utils import get_logger
from unet3d.utils import get_number_of_learnable_parameters


def _arg_parser():
    parser = argparse.ArgumentParser(description='UNet3D training')
    parser.add_argument('--config-dir', help='config directory')
    parser.add_argument('--checkpoint-dir', help='checkpoint directory')
    parser.add_argument('--in-channels', default=1, type=int,
                        help='number of input channels')
    parser.add_argument('--out-channels', default=6, type=int,
                        help='number of output channels')
    parser.add_argument('--interpolate',
                        help='use F.interpolate instead of ConvTranspose3d',
                        action='store_true')
    parser.add_argument('--batchnorm',
                        help='use BatchNorm3d before nonlinearity',
                        action='store_true')
    parser.add_argument('--epochs', default=100, type=int,
                        help='max number of epochs')
    parser.add_argument('--learning-rate', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                        help='weight decay')
    parser.add_argument('--validate-after-iters', default=100, type=int,
                        help='how many iterations between validations')
    parser.add_argument('--log-after-iters', default=100, type=int,
                        help='how many iterations between tensorboard logging')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    return parser


def _create_model(in_channels, out_channels, interpolate=False,
                  final_sigmoid=True, batch_norm=True):
    return UNet3D(in_channels, out_channels, interpolate, final_sigmoid,
                  batch_norm)


def _get_loaders(config_dir, logger):
    train_config_file = os.path.join(config_dir, 'data_config.yml')
    logger.info(f"Loading training data loader from {train_config_file}")
    train_loader = get_primordia_loaders(train_config_file)

    validation_config_file = os.path.join(config_dir, 'validation_config.yml')
    logger.info(f"Loading validation data loader from {validation_config_file}")
    validation_loader = get_primordia_loaders(validation_config_file)

    return {
        'train': train_loader,
        'val': validation_loader
    }


def _create_criterions(final_sigmoid):
    if final_sigmoid:
        loss_criterion = nn.BCELoss()
    else:
        loss_criterion = nn.CrossEntropyLoss()
    error_criterion = DiceCoefficient()
    return error_criterion, loss_criterion


def _create_optimizer(args, model):
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=weight_decay)
    return optimizer


def main():
    parser = _arg_parser()
    logger = get_logger('UNet3DTrainer')
    # Get device to train on
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    logger.info(args)
    # Treat different output channels as different binary segmentation masks
    # instead of treating each channel as a mask for a different class
    out_channels_as_classes = False
    final_sigmoid = not out_channels_as_classes
    model = _create_model(args.in_channels, args.out_channels,
                          interpolate=args.interpolate,
                          final_sigmoid=final_sigmoid,
                          batch_norm=args.batchnorm)

    # Log the number of learnable parameters
    logger.info(
        f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create loss criterion and error metric
    error_criterion, loss_criterion = _create_criterions(final_sigmoid)

    # Get data loaders
    loaders = _get_loaders(args.config_dir, logger)

    # Create the optimizer
    optimizer = _create_optimizer(args, model)

    if args.resume:
        trainer = UNet3DTrainer.from_checkpoint(args.resume, model, optimizer,
                                                loss_criterion, error_criterion,
                                                loaders,
                                                logger=logger)
    else:
        trainer = UNet3DTrainer(model, optimizer, loss_criterion,
                                error_criterion,
                                device, loaders, args.checkpoint_dir,
                                validate_after_iters=args.validate_after_iters,
                                log_after_iters=args.log_after_iters,
                                logger=logger)

    trainer.fit()


if __name__ == '__main__':
    main()
