import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
class LSTM_V(object):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )

    def run(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: Variable
        :param x_len: numpy list
        :return:
        """
        batch_size = x.size(0)
        """sort"""
        x_sort_idx = np.argsort(-x_len)
        x_unsort_idx = torch.LongTensor(np.argsort(x_sort_idx))
        x_len = x_len[x_sort_idx]
        x = x[torch.LongTensor(x_sort_idx)]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        """process using RNN"""
        hidden = self.init_hidden(batch_size, self.num_layers, self.hidden_size)
        
        out_pack, (ht, ct) = self.LSTM(x_emb_p, hidden)
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            """unsort: out c"""
            out = out[x_unsort_idx]
            ct = torch.transpose(ct, 0, 1)[
                x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
            ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)


    def init_hidden(self, batch_size, num_layers, hidden_size):
        weight = next(self.LSTM.parameters()).data

        return (Variable(weight.new(num_layers*2, batch_size, hidden_size).zero_()),
                Variable(weight.new(num_layers*2, batch_size, hidden_size).zero_()))