import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet3d.model import UNet3D
from unet3d.trainer import UNet3DTrainer
from unet3d.utils import DiceCoefficient
from unet3d.utils import Random3DDataset
from unet3d.utils import get_logger


class TestUNet3DTrainer(object):
    def test_single_epoch(self, tmpdir, capsys):
        with capsys.disabled():
            # get device to train on
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else 'cpu')

            # Treat different output channels as different segmentation mask
            # Ground truth data should have the same number of channels in this case
            out_channels_as_classes = False
            batch_norm = True
            model = self._load_model(not out_channels_as_classes, batch_norm)
            # Create criterion
            if out_channels_as_classes:
                loss_criterion = nn.CrossEntropyLoss()
            else:
                loss_criterion = nn.BCELoss()

            error_criterion = DiceCoefficient()

            loaders = self._get_loaders()

            learning_rate = 1e-4
            weight_decay = 0.0005
            optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                                   weight_decay=weight_decay)

            logger = get_logger('UNet3DTrainer')
            trainer = UNet3DTrainer(model, optimizer, loss_criterion,
                                    error_criterion,
                                    device, loaders, tmpdir,
                                    max_num_epochs=1,
                                    log_after_iters=2,
                                    validate_after_iters=2,
                                    logger=logger)

            trainer.fit()

            # test loading the trainer from the checkpoint
            UNet3DTrainer.from_checkpoint(
                os.path.join(tmpdir, 'last_checkpoint.pytorch'),
                model, optimizer, loss_criterion, error_criterion, loaders,
                logger=logger)

    def _load_model(self, final_sigmoid, batch_norm):
        in_channels = 1
        out_channels = 2
        # use F.interpolate for upsampling
        interpolate = True
        return UNet3D(in_channels, out_channels, interpolate,
                      final_sigmoid, batch_norm)

    def _get_loaders(self):
        # when using ConvTranspose3d, make sure that dimensions can be divided by 16
        train_dataset = Random3DDataset(4, (32, 64, 64), 2)
        val_dataset = Random3DDataset(1, (32, 64, 64), 2)

        return {
            'train': DataLoader(train_dataset, batch_size=1, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=1, shuffle=True)
        }
