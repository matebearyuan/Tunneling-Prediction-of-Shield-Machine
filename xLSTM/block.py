"""
xLSTM Block Implementation

This module implements the xLSTM block as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The xLSTM block combines either sLSTM or mLSTM with layer normalization,
residual connections, and additional linear projections.

Author: Mudit Bhargava
Date: June 2024
"""

import torch
import torch.nn as nn
from .slstm import sLSTM
from .mlstm import mLSTM


class xLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, lstm_type="slstm"):
        super(xLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm_type = lstm_type

        if lstm_type == "slstm":
            self.lstm = sLSTM(input_size, hidden_size, num_layers, dropout)
        elif lstm_type == "mlstm":
            self.lstm = mLSTM(input_size, hidden_size, num_layers, dropout)
        else:
            raise ValueError(f"Invalid LSTM type: {lstm_type}")

        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.input_proj = nn.Linear(input_size, hidden_size)  # 新增输入映射

    def forward(self, input_seq, hidden_state=None):
        lstm_output, hidden_state = self.lstm(input_seq, hidden_state)
        output = self.activation(lstm_output)
        output = self.norm(output)
        output = self.proj(output)

        # 将 input_seq 映射到 hidden_size，用于残差连接
        input_proj = self.input_proj(input_seq)
        output = self.dropout_layer(output + input_proj)  # 残差连接
        return output, hidden_state
