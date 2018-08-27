import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        pass


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic;
    make sure to use complementary kernel in the decoder path) followed by
    2 convolution layers (conv+BatchNorm+ReLU) with the number of output channels
    out_channels / 2 and out_channels respectively.
    """

    def __init__(self, in_channels, out_channels, max_pool=True,
                 max_pool_kernel_size=(2, 2, 2)):
        super(Encoder, self).__init__()
        self.max_pool = nn.MaxPool3d(
            kernel_size=max_pool_kernel_size) if max_pool else None
        self.double_conv = DoubleConv(in_channels, out_channels)

        nn.Seq
