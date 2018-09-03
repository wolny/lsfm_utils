import logging

import sys

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
from unet3d import UNet3D
from utils import save_checkpoint, load_checkpoint, RunningAverage, \
    ComposedLoss, MeanIoU


class UNet3DTrainer:
    """3D UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        loss_criterion (callable): loss function
        error_criterion (callable): used to compute training/validation error
            best checkpoint is based on the result of this function
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        best_val_error (float): best validation error so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
    """

    def __init__(self, model, optimizer, loss_criterion, error_criterion,
                 device, loaders, checkpoint_dir,
                 max_num_epochs=100, max_num_iterations=1e5,
                 validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, best_val_error=float('-inf'),
                 num_iterations=0, num_epoch=0, logger=None):
        if logger is None:
            self.logger = self._get_logger()
        else:
            self.logger = logger

        # IMPORTANT
        # in our case evaluate on the whole validation set (validate_iters=None)
        # in order to get comparable results, since the ground truth is not ideal

        self.logger.debug(f"Sending the model to '{device}'")
        self.model = model.to(device)
        self.logger.debug(model)

        self.optimizer = optimizer
        self.loss_criterion = loss_criterion
        self.error_criterion = error_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.best_val_error = best_val_error
        self.writer = SummaryWriter(
            log_dir=os.path.join(checkpoint_dir, 'logs'))

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch

    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, optimizer, loss_criterion,
                        error_criterion, loaders):
        logger = cls._get_logger()
        logger.info(f"Loading checkpoint '{checkpoint_path}'...")
        state = load_checkpoint(checkpoint_path, model, optimizer)
        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. Best val error: {state['best_val_error']}")
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        return cls(model, optimizer, loss_criterion, error_criterion,
                   torch.device(state['device']), loaders,
                   checkpoint_dir, best_val_error=state['best_val_error'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'], logger=logger)

    def fit(self):
        for num_epoch in range(self.num_epoch, self.max_num_epochs):
            self.adjust_learning_rate(num_epoch)

            # train for one epoch
            self.train(self.loaders['train'], num_epoch)

            if self.max_num_iterations < self.num_iterations:
                self.logger.info(
                    f'Maximum number of iterations {max_num_iterations} exceeded. Finishing training...')

    def adjust_learning_rate(self, num_epoch):
        pass

    def train(self, train_loader, num_epoch):
        train_losses = RunningAverage()
        train_errors = RunningAverage()

        self.model.train()

        for i, (input, target) in enumerate(train_loader):
            self.logger.info(
                f'Training iteration {self.num_iterations}. Batch {i}. Epoch [{num_epoch}/{self.max_num_epochs - 1}]')

            input, target = input.to(device), target.to(device)

            # forward pass
            output = self.model(input)

            # mesure training loss and error
            loss = self.loss_criterion(output, target)
            error = self.error_criterion(output, target)

            train_losses.update(loss.item(), input.size(0))
            train_errors.update(error.item(), input.size(0))

            # compute gradients and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.num_iterations += 1

            if self.num_iterations % self.log_after_iters == 0:
                # log stats, params and images
                self._log_stats('train', train_losses.avg, train_errors.avg)
                self._log_params()
                # TODO: images

            if self.num_iterations % self.validate_after_iters == 0:
                # evaluate on validation set
                val_error = self.validate(self.loaders['val'])

                # remember best validation metric
                is_best = self._is_best_val_error(val_error)

                # save checkpoint
                self._save_checkpoint(is_best, num_epoch)

    def validate(self, val_loader):
        self.logger.info('Validating...')

        val_losses = RunningAverage()
        val_errors = RunningAverage()

        self.model.eval()
        try:
            with torch.no_grad():
                for i, (input, target) in enumerate(val_loader):
                    self.logger.info(f'Validation iteration {i}')
                    input, target = input.to(device), target.to(device)

                    # forward pass
                    output = self.model(input)

                    loss = self.loss_criterion(output, target)
                    error = self.error_criterion(output, target)

                    val_losses.update(loss.item(), input.size(0))
                    val_errors.update(error.item(), input.size(0))

                self._log_stats('val', val_losses.avg, val_errors.avg)
                self.logger.info(f'Validation finished. Error {val_errors.avg}')
                return val_errors.avg
        finally:
            self.model.train()

    def _is_best_val_error(self, val_error):
        is_best = val_error > self.best_val_error
        if is_best:
            self.logger.info(
                f'Saving new best validation error: {val_error}')
        self.best_val_error = max(val_error, self.best_val_error)
        return is_best

    def _save_checkpoint(self, is_best, num_epoch):
        save_checkpoint({
            'epoch': num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_val_error': self.best_val_error,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device)
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)

    @staticmethod
    def _get_logger():
        logger = logging.getLogger('UNet3DTrainer')
        logger.setLevel(logging.DEBUG)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        return logger

    def _log_stats(self, phase, loss_avg, error_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_error_avg': error_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(),
                                      self.num_iterations)
            self.writer.add_histogram(name + '/grad',
                                      value.grad.data.cpu().numpy(),
                                      self.num_iterations)


def load_model():
    in_channels = 1
    out_channels = 1
    # use F.interpolate for upsampling
    interpolate = True
    return UNet3D(in_channels, out_channels, interpolate)


class Random3DDataset(Dataset):
    def __init__(self, N, size, out_channels):
        raw_dims = (N, 1) + size
        labels_dims = (N, out_channels) + size
        self.raw = torch.randn(raw_dims)
        self.labels = torch.empty(labels_dims, dtype=torch.float).random_(2)

    def __len__(self):
        return self.raw.size(0)

    def __getitem__(self, idx):
        return self.raw[idx], self.labels[idx]


def get_loaders():
    # when using ConvTranspose3d, make sure that dimensions can be divided by 16
    train_dataset = Random3DDataset(4, (32, 64, 64), 1)
    val_dataset = Random3DDataset(1, (32, 64, 64), 1)

    return {
        # tensorboard logger

        'train': DataLoader(train_dataset, batch_size=1,
                            shuffle=True),
        'val': DataLoader(val_dataset, batch_size=1,
                          shuffle=True)
    }


if __name__ == '__main__':
    model = load_model()
    # get device to train on
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # Treat different output channels as different segmentation mask
    # Ground truth data should have the same number of channels in this case
    out_channels_as_classes = False
    # Create criterion
    if out_channels_as_classes:
        loss_criterion = nn.CrossEntropyLoss()
    else:
        loss_criterion = ComposedLoss(nn.Sigmoid(), nn.BCELoss())

    error_criterion = MeanIoU()

    loaders = get_loaders()

    learning_rate = 1e-4
    weight_decay = 0.0005
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=weight_decay)

    trainer = UNet3DTrainer(model, optimizer, loss_criterion, error_criterion,
                            device, loaders, './checkpoints', log_after_iters=8,
                            validate_after_iters=8)

    # trainer = UNet3DTrainer.from_checkpoint(
    #     './checkpoints/last_checkpoint.pytorch',
    #     model, optimizer, loss_criterion, error_criterion, loaders)

    trainer.fit()
