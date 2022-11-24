import math

import numpy as np
#import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F

torch.set_printoptions(precision=20)


class MyModel(nn.Module):
    def __init__(self, input_channel=1, kernel_size=1, cuda=False):
        super(MyModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, out_channels=input_channel * 2, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(input_channel * 2),
            nn.Conv1d(input_channel * 2, out_channels=input_channel, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(input_channel)
        )

        self.MyAttention = SelfAttention(input_channel, input_channel, 0.2)

        self.conv0 = nn.Conv1d(input_channel, out_channels=input_channel, kernel_size=1, padding=0,
                               padding_mode='circular')
        self.conv1 = nn.Conv1d(input_channel, out_channels=input_channel, kernel_size=1, padding=49, stride=2,
                               padding_mode='replicate')
        self.conv2 = nn.Conv1d(input_channel, out_channels=input_channel, kernel_size=1, padding=97, stride=3,
                               padding_mode='replicate')

        if cuda:
            self.conv = self.conv.cuda()
            self.conv0 = self.conv0.cuda()
            self.conv1 = self.conv1.cuda()
            self.conv2 = self.conv2.cuda()
            self.MyAttention = self.MyAttention.cuda()

    def forward(self, x):
        x = self.conv(x).transpose(2, 1) #channel * timepoint -> timepoint * channel
        x, x2 = self.MyAttention(x)
        x = x.transpose(2, 1)
        y = self.conv0(x)
        return y, x2


class SelfAttention(nn.Module):

    def __init__(self, channel_size, num_heads, dropout_prob):
        super(SelfAttention, self).__init__()
        if channel_size % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (channel_size, num_heads))

        self.num_heads = num_heads
        self.head_size = int(channel_size / num_heads)  # each head attention dimension
        self.all_head_size = int(self.num_heads * self.head_size)
        # all_head_size = channel size, the dimension of self-attention input and output does not change before and after

        # query, key, value
        self.query = nn.Linear(channel_size, self.all_head_size)
        self.key = nn.Linear(channel_size, self.all_head_size)
        self.value = nn.Linear(channel_size, self.all_head_size)

        # dropout
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        # INPUT:  x'shape = [bs, seqlen, channel size]
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)  # [bs, seqlen, num_heads, all_head_size]
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)  # [bs, num_heads, seqlen, all_head_size]

    def forward(self, hidden_states):
        #print("selfAttention(q): ", hidden_states.shape)
        mixed_query_layer = self.query(hidden_states)  # [bs, seqlen, channel size]
        mixed_key_layer = self.key(hidden_states)  # [bs, seqlen, channel size]
        mixed_value_layer = self.value(hidden_states)  # [bs, seqlen, channel size]
        #print("selfAttention(q): ", mixed_query_layer.shape)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [bs, num_heads, seqlen, all_head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [bs, num_heads, seqlen, all_head_size]

        #print("selfAttention(q): ", query_layer.permute(0, 3, 1, 2).shape)
        #print("selfAttention(k): ", key_layer.permute(0, 3, 2, 1).shape)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        #attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = torch.matmul(query_layer.permute(0, 3, 1, 2), key_layer.permute(0, 3, 2, 1))
        attention_scores = attention_scores / math.sqrt(self.head_size)  # [bs, num_heads, seqlen, seqlen]
        # 除以根號head attention的數量，防止分數過大，過大會導致softmax之後非0即1

        #********
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, num_heads, seqlen, seqlen]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        #print("selfAttention(v): ", value_layer.permute(0, 2, 1, 3).shape)
        # [bs, num_heads, seqlen, seqlen]*[bs, num_heads, seqlen, all_head_size] = [bs, num_heads, seqlen, all_head_size]
        #context_layer = torch.matmul(attention_probs, value_layer)  # [bs, num_heads, seqlen, all_head_size]
        context_layer = torch.matmul(attention_probs, value_layer.permute(0, 2, 1, 3))
        #print("selfAttention(c): ", context_layer.shape)
        context_layer = context_layer.permute(0, 1, 2, 3).contiguous()  # [bs, seqlen, num_heads, all_head_size]
        #print("selfAttention(c): ", context_layer.shape)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [bs, seqlen, hidden size]
        #print("selfAttention: ", new_context_layer_shape)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs  # [bs, seqlen, hidden size]


if __name__ == '__main__':
    inputs = torch.rand(1, 64, 3350)
    model = MyModel(input_channel=64)
    output, matrix_A = model(inputs)
    print(output.shape)
    print(matrix_A.shape)
