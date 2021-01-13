"""handles fast shot implementation"""
from torch import nn
import torch.nn.functional as F


class FastShot(nn.Module):
    """
    Attempted quick reimplemntation of the paper
    does not adjust for size
    """
    def __init__(self):

        super(FastShot, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, (8, 30, 30), stride=1)
        self.conv2 = nn.Conv3d(16, 24, (6, 12, 12), stride=1)
        self.conv3 = nn.Conv3d(24, 32, (4, 6, 6), stride=1)
        self.conv4 = nn.Conv3d(32, 12, (4, 1, 1), stride=1)
        self.softmax_layer = nn.Conv3d(12, 2, (1, 1, 1), stride=1)

    def _check_dims(self, prev_conv, input_channel, output_channel,
                    kernel_shape):
        """
        checks if the given dimensions will be appropriate for the next one
        :param prev_conv: nn convolution
        :param input_channel: input channel size
        :param output_channel: output channel size
        :param kernel_shape: d, h, w
        :return:
            bool
        """
        return True

    def _output_correct_dim(self, d, h, w, kernel_size, stride):
        """
        given previous convolution size what should next one be
        next h w d of next convolution
        :param prev_conv: nn convolution

        :return:
            tuple
        """

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = F.log_softmax(self.softmax_layer(x), dim=1)

        return x
