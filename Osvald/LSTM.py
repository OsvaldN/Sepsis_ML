#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
# for hotfix to pack_padded_sequence
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence


class lstm(nn.Module):
    '''
    lstm prototype
    input -> [B, n, 40] physiological variable time series tensor
          -> [B, n, 80] if missing markers are included
    output -> [B, n] sepsis label tensor
    '''
    def __init__(self, embedding, hidden_size, inp=40, fcl=32, num_layers=2, batch_size=1, fcl_out=False, embed=False, droprate=0.5):
        super(lstm, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm_layers = num_layers
        self.embed = embed

        self.inp = nn.Linear(inp, embedding) # input embedding - can be changed, potentially to small TCN
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