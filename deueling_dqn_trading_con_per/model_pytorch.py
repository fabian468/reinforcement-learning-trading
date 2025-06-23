# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 10:18:25 2025

@author: fabia
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 10:18:25 2025

@author: fabia
"""
import torch.nn as nn
import torch.nn.functional as F

from NoisyLinear import NoisyLinear

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_space):
        super(DuelingDQN, self).__init__()

        self.layer_norm = nn.LayerNorm(state_size)

        # Capas densas principales
        self.fc1 = nn.Linear(state_size, 512)
        self.ln1 = nn.LayerNorm(512)
        self.dropout1 = nn.Dropout(0.05)

        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.dropout2 = nn.Dropout(0.05)

        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)
        self.dropout3 = nn.Dropout(0.05)

        # Streams: value y advantage con NoisyLinear
        self.value_stream = nn.Linear(128, 64)
        self.value_ln = nn.LayerNorm(64)
        self.value_dropout = nn.Dropout(0.05)
        self.value = NoisyLinear(64, 1)

        self.advantage_stream = nn.Linear(128, 128)
        self.advantage_ln = nn.LayerNorm(128)
        self.advantage_dropout = nn.Dropout(0.05)
        self.advantage = NoisyLinear(128, action_space)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and not isinstance(m, NoisyLinear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer_norm(x)

        x = F.leaky_relu(self.ln1(self.fc1(x)), negative_slope=0.1)
        x = self.dropout1(x)

        x = F.leaky_relu(self.ln2(self.fc2(x)), negative_slope=0.1)
        x = self.dropout2(x)

        x = F.leaky_relu(self.ln3(self.fc3(x)), negative_slope=0.1)
        x = self.dropout3(x)

        value_stream = F.leaky_relu(self.value_ln(self.value_stream(x)), negative_slope=0.1)
        value_stream = self.value_dropout(value_stream)
        value = self.value(value_stream)

        advantage_stream = F.leaky_relu(self.advantage_ln(self.advantage_stream(x)), negative_slope=0.1)
        advantage_stream = self.advantage_dropout(advantage_stream)
        advantage = self.advantage(advantage_stream)

        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)

        return q_values

    def reset_noise(self):
        self.value.reset_noise()
        self.advantage.reset_noise()
