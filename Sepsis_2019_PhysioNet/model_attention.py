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

#TODO: add attention

class Encoder(nn.Module):
    '''
    Encoder prototype
    input -> [B, n, 40] physiological variable time series tensor
    output -> [B, n, hidden_size] encoded tensor
    '''
    def __init__(self, embedding, hidden_size, num_layers=2, batch_size=1, embed=False, droprate=0.5):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm_layers = num_layers
        self.embed = embed

        self.inp = nn.Linear(40, embedding) # input embedding - can be changed, potentially to small TCN
        self.drop1 = nn.Dropout(p=droprate)
        self.rnn = nn.LSTM(embedding, hidden_size, num_layers=num_layers, batch_first=True, dropout=droprate) # RNN structure
        self.drop2 = nn.Dropout(p=droprate)

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
        #X = self.drop1(X)
        X = hotfix_pack_padded_sequence(X, seq_len, batch_first=True, enforce_sorted=False)
        #X = torch.nn.utils.rnn.pack_padded_sequence(X, seq_len, batch_first=True, enforce_sorted=False)
        X, self.hidden = self.rnn(X, self.hidden)
        X, hidden = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True, padding_value=-1, total_length=max_len)
        #X = self.drop2(X)
        return X.squeeze(), self.hidden


class Decoder(nn.Module):
    def __init__(self, batch_size, hidden_size=64, droprate=0.5):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.droprate = droprate
        self.batch_size = batch_size

        self.attn = nn.Linear(hidden_size, hidden_size)
        # hc: [hidden, context]
        self.Whc = nn.Linear(hidden_size * 2, hidden_size)
        # s: softmax
        self.Ws = nn.Linear(hidden_size, 1)

        self.hidden = torch.randn(self.batch_size, self.hidden_size)

    def forward(self, encoder_outputs, hidden):
        #TODO: fix output to sequence rather than single output
        attn_prod = torch.bmm(self.attn(hidden).unsqueeze(1), encoder_outputs.permute(0,2,1))
        attn_weights = F.softmax(attn_prod, dim=1)
        context = torch.bmm(attn_weights, encoder_outputs).squeeze()

        # hc: [hidden: context]
        hc = torch.cat([hidden, context], dim=1)
        output = self.Ws(F.relu(self.Whc(hc)))

        return output, hidden, attn_weights

class LSTM_attn(nn.Module):
    '''
    lstm w/ attention
    '''
    def __init__(self, embedding, hidden_size, num_layers=2, batch_size=1, embed=False, droprate=0.5):
        super(LSTM_attn, self).__init__()
        self.encoder = Encoder(embedding, hidden_size, num_layers, batch_size, embed, droprate)
        self.decoder = Decoder(batch_size=batch_size)
    
    def forward(self,X, seq_len, max_len):
        outputs = self.encoder(X, seq_len, max_len)
        outputs = self.decoder(outputs[0], outputs[1][0][1])
        return outputs
'''
class Attention(nn.Module):
    """
    args -> dim(int): The number of expected features in the output

    Inputs -> output (B, out_len, hidden_dim): tensor containing the output features from the decoder.
           -> context (B, in_len, hidden_dim): tensor containing features of the encoded input sequence.

    Outputs -> output (B, out_len, hidden_dim): tensor containing the attended output features from the decoder.
            -> attn (B, out_len, in_len): tensor containing attention weights.

    Examples::
         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)
    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn

class Decoder(nn.Module):
    ''''''
    decoder prototype
    input -> [B, n, hidden_size] encoded tensor
    output -> [B, n] spesis prediction tensor
    ''''''
    def __init__(self, hidden_size, batch_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

    #def forward_step(self,): 
'''

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




