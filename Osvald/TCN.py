#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    '''
    I think this removes padding but not sure
    '''
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # remove 
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(n_inputs)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.ReLU1 = nn.SELU()
        self.dropout1 = nn.Dropout(dropout)

        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.ReLU2 = nn.SELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.bn1, self.conv1, self.chomp1, self.ReLU1, self.dropout1,
                                 self.bn2, self.conv2, self.chomp2, self.ReLU2, self.dropout2)

        # Lower Dimension of output data if necessary
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.ReLU = nn.SELU()

        for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.ReLU(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x) 

class TCN(nn.Module):
    '''
    input size: dim of data w/in each timestep
    output size: N_classes
    num_channels: list of channel widtch for each block
                  ex. [60, 40, 10] for a 3 block network
                  note: n_blocks affects receptive field ( i think 2 ^ n_layers)

    '''
    def __init__(self, input_size, output_size, num_channels, fcl, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        #maybe add lstm here?
        self.linear = nn.Linear(num_channels[-1], fcl)
        if dropout:
            self.drop = torch.nn.Dropout(p=dropout)
        else:
            self.drop = False
        self.out = nn.Linear(fcl, output_size)
        #self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs dims (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        y1 = y1.permute(0,2,1)
        out = self.linear(y1)
        if self.drop:
            out = self.drop(out)
        out = self.out(out)
        #out = self.linear(y1[:, :, -1]) # and remove permute if single output
        return out.squeeze()
