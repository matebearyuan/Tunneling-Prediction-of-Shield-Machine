"""
xLSTM: Extended Long Short-Term Memory Model

This module implements the xLSTM model as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The xLSTM model combines sLSTM and mLSTM blocks in a residual architecture
to achieve state-of-the-art performance on various language modeling tasks.

Author: Mudit Bhargava
Date: June 2024
"""

import torch
import torch.nn as nn
from .block import xLSTMBlock


class xLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_blocks, dropout=0.0, lstm_type="slstm"):
        super(xLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lstm_type = lstm_type

        # 修改输入映射层：将输入特征的维度从 vocab_size 映射到 embedding_size
        self.input_proj = nn.Linear(vocab_size, embedding_size)

        self.blocks = nn.ModuleList([xLSTMBlock(embedding_size, hidden_size, num_layers, dropout, lstm_type)
                                     for _ in range(num_blocks)])

        # 输出层调整为正确的维度
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, input_seq, hidden_states=None):
        # 确保输入数据是 (batch_size, seq_length, input_size) 的 3D 张量
        batch_size, seq_length, _ = input_seq.size()

        input_seq = self.input_proj(input_seq)  # 映射到 embedding_size
        input_seq = input_seq.view(batch_size, seq_length, self.embedding_size)  # 转换为适当的维度

        if hidden_states is None:
            hidden_states = [None] * self.num_blocks

        output_seq = input_seq
        for i, block in enumerate(self.blocks):
            output_seq, hidden_states[i] = block(output_seq, hidden_states[i])

        output_seq = self.output_layer(output_seq[:, -1, :])  # 取最后一个时间步的输出
        return output_seq, hidden_states




