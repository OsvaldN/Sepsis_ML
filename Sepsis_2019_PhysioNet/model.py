#!/usr/bin/env python

import numpy as np
import os, sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import math
# for hotfix to pack_padded_sequence
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence


class lstm(nn.Module):
    '''
    lstm prototype
    input -> [B, n, 40] physiological variable time series tensor
    output -> [B, n] sepsis label tensor
    '''
    def __init__(self, embedding, hidden_size, fcl=32, num_layers=2, batch_size=1, fcl_out=False, embed=False, droprate=0.5):
        super(lstm, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm_layers = num_layers
        self.embed = embed

        self.inp = nn.Linear(40, embedding) # input embedding - can be changed, potentially to small TCN
        self.drop1 = nn.Dropout(p=droprate)
        self.rnn = nn.LSTM(embedding, hidden_size, num_layers=num_layers, batch_first=True, dropout=droprate) # RNN structure
        self.drop2 = nn.Dropout(p=droprate)

        self.fcl_out = fcl_out
        if self.fcl_out:
            self.fcl = nn.Linear(hidden_size, fcl)
            self.drop3 = nn.Dropout(p=droprate)
            self.out = nn.Linear(fcl, 1) 
        else:
            self.out = nn.Linear(hidden_size, 1) # output linear

        for m in self.modules():
           if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def init_hidden(self):
        hidden_a = torch.randn(self.lstm_layers, self.batch_size, self.hidden_size)
        hidden_b = torch.randn(self.lstm_layers, self.batch_size, self.hidden_size)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)
    
    def forward(self, X, seq_len, max_len, hidden_state=None): 
        self.hidden = self.init_hidden()
        if self.embed:
            X = self.inp(X)
        X = self.drop1(X)
        X = hotfix_pack_padded_sequence(X, seq_len, batch_first=True, enforce_sorted=False)
        #X = torch.nn.utils.rnn.pack_padded_sequence(X, seq_len, batch_first=True, enforce_sorted=False)
        X, self.hidden = self.rnn(X, self.hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True, padding_value=-1, total_length=max_len)
        X = self.drop2(X)
        if self.fcl_out:
            X = self.drop3(self.fcl(X))
        X = self.out(X)
        return X.squeeze()


def hotfix_pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    '''
    #TODO: if ever fixed just go back to original
    GPU errors with orig func:
    torch.nn.utils.rnn.pack_padded_sequence()
    this fix was provided on pytorch board
    ''' 
    lengths = torch.as_tensor(lengths, dtype=torch.int64)
    lengths = lengths.cpu()
    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)

    data, batch_sizes = \
        torch._C._VariableFunctions._pack_padded_sequence(input, lengths, batch_first)
    return PackedSequence(data, batch_sizes, sorted_indices)







'''TCN'''

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
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.bn1, self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.bn2, self.conv2, self.chomp2, self.relu2, self.dropout2)

        # Lower Dimension of output data if necessary
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

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
        return self.relu(out + res)


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
