"""
This is the module where the model is defined. It uses the nn.Module as a backbone to create the network structure
"""

import math
import numpy as np

# Pytorch module
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import pow, add, mul, div, sqrt, square

class Forward(nn.Module):
    def __init__(self, flags):
        super(Forward, self).__init__()

        """
        General FC layer definitions:
        """
        # Linear Fully Connected Layer and Batch_norm Layer definitions here
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear[0:-1]):
            self.linears.append(nn.Linear(fc_num, flags.linear[ind + 1], bias=True))
            self.bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1], track_running_stats=True, affine=True))

        # Convolutional Layer definitions here
        self.use_conv = flags.use_conv
        self.convs = nn.ModuleList([])
        in_channel = 1  # Initialize the in_channel number
        for ind, (out_channel, kernel_size, stride) in enumerate(zip(flags.conv_out_channel,
                                                                     flags.conv_kernel_size,
                                                                     flags.conv_stride)):
            if stride == 2:  # We want to double the number
                pad = int(kernel_size / 2 - 1)
            elif stride == 1:  # We want to keep the number unchanged
                pad = int((kernel_size - 1) / 2)
            else:
                Exception("Now only supports stride = 1 or 2")

            self.convs.append(nn.ConvTranspose1d(in_channel, out_channel, kernel_size,
                                                 stride=stride, padding=pad))  # To make sure L_out double each time
            in_channel = out_channel  # Update the out_channel

        self.convs.append(nn.Conv1d(in_channel, out_channels=1, kernel_size=1, stride=1, padding=0))


    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input layer (Since this is a forward network)
        :return: out: output
        """
        out = G

        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):

            if ind < len(self.linears) - 0:
                out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear
            else:
                out = bn(fc(out))

        if self.use_conv:
            out = out.unsqueeze(1)  # Add 1 dimension to get N,L_in, H
            # For the conv part
            for ind, conv in enumerate(self.convs):
                out = conv(out)

            out = out.squeeze()

        return out
