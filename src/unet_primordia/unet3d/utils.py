import os
import shutil

import torch
import torch.nn as nn


def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best validation error so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])

    return state


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


class ComposedLoss(nn.Module):
    """Helper class for composed loss functions.

    Args:
        input_func (nn.Module): element-wise function applied on the input before
            passing it to the output
        loss (nn.Module): loss function to be applied on the transformed input and target

    Example:
        ```
        loss = ComposedLoss(nn.Sigmoid(), nn.BCELoss())
        output = loss(input, target)
        ```
        would be equivalent to:
        ```
        loss = nn.BCELoss()
        output = loss(F.sigmoid(input), target)
        ```
    """

    def __init__(self, input_func, loss):
        super(ComposedLoss, self).__init__()
        self.input_func = input_func
        self.loss = loss

    def forward(self, input, target):
        return self.loss(self.input_func(input), target)


class MeanIoU:
    def __init__(self, num_of_classes=2):
        self.num_of_classes = num_of_classes

    def __call__(self, prediction, target):
        # TODO: implement
        return torch.rand(1)