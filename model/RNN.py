#import IPython
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from model.LSTM_V import LSTM_V
from utils.dataset import MyDataSet


class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.event_embedding = nn.Embedding(config['event_vocab_size'], config['embedding_dim'])
        self.time_embedding = nn.Embedding(config['time_vocab_size'], config['embedding_dim'])
        self.type_embedding = nn.Embedding(config['type_vocab_size'], config['embedding_dim'])

        self.log_embedding_dim = config['embedding_dim'] * 5

        self.log_lstm = nn.LSTM(
            self.log_embedding_dim, config['hidden_dim'], config['num_layers'],
            bidirectional=True, batch_first=True)
        self.agg_to_hidden = nn.Linear(config['agg_input_dim'], config['agg_hidden_dim'])

        self.dropout = nn.Dropout(config['drop_out'])
        self.output = nn.Linear((config['agg_hidden_dim']+self.config['hidden_dim']*2), 2)
        self.pool = nn.MaxPool2d((config['max_len'],1))
        self.relu = nn.ReLU()

    def forward(self, agg_input, event_0, event_1, event_2, hour, type, len):

        batch_size = agg_input.size(0)
        agg_input = agg_input.float()
        agg_hidden = self.agg_to_hidden(agg_input)
        log_embedding = self.log_repr( event_0, event_1, event_2, hour, type)
        hidden = self.init_hidden(batch_size, self.config['num_layers'], self.config['hidden_dim'])
        log_rnn_outputs, _ = self.log_lstm(log_embedding, hidden)

        log_rnn_outputs = log_rnn_outputs.contiguous()
        log_rnn_output = self.pool(log_rnn_outputs.view(batch_size, 1, self.config['max_len'], -1)).view(
            batch_size, -1)
        mix_repr = torch.cat((log_rnn_output,agg_hidden),dim=1)
        output = self.output(self.dropout(mix_repr))
        return output

    def init_hidden(self, batch_size, num_layers, hidden_size):
        weight = next(self.parameters()).data

        return (Variable(weight.new(num_layers*2, batch_size, hidden_size).zero_()),
                Variable(weight.new(num_layers*2, batch_size, hidden_size).zero_()))

    def init_weights(self):
        self.event_embedding.weight.data.normal_()
        self.time_embedding.weight.data.normal_()
        self.type_embedding.weight.data.normal_()

        self.output.weight.data.uniform_(-0.1, 0.1)
        self.output.bias.data.fill_(0)
        self.agg_to_hidden.weight.data.uniform_(-0.1, 0.1)
        self.agg_to_hidden.bias.data.fill_(0)

        for name, param in self.log_lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)
            elif 'weight' in name:
                nn.init.xavier_normal(param)

    def log_repr(self,event_0, event_1, event_2, hour, type):

        event_0_embedding = self.event_embedding(event_0)
        event_1_embedding = self.event_embedding(event_1)
        event_2_embedding = self.event_embedding(event_2)
        event_embedding = torch.cat((event_0_embedding, event_1_embedding, event_2_embedding), dim=-1)
        hour_embedding = self.time_embedding(hour)
        type_embedding = self.type_embedding(type)
        log_embedding = torch.cat((event_embedding, hour_embedding, type_embedding), dim=-1)

        return log_embedding

