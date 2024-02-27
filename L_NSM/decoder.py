# This file contains gating and component network


import torch
import torch.nn as nn
import torch.nn.functional  as F
import numpy as np

rng = np.random.RandomState(23456)

class PredictionNet(nn.Module):
    def __init__(self, n_input, n_output, n_expert_weights, h, drop_prob=0.3, rng=rng):
        super(PredictionNet, self).__init__()
        self.n_expert_weights = n_expert_weights
        self.n_input = n_input
        self.n_output = n_output
        self.h = h

        self.register_parameter(name='expert_weights_fc0',
                                param=self.initial_alpha((n_expert_weights, h, n_input), rng))
        self.register_parameter(name='expert_weights_fc1', param=self.initial_alpha((n_expert_weights, h, h), rng))
        self.register_parameter(name='expert_weights_fc2',
                                param=self.initial_alpha((n_expert_weights, n_output, h), rng))
        self.register_parameter(name='expert_bias_fc0', param=nn.Parameter(torch.zeros((n_expert_weights, h))))
        self.register_parameter(name='expert_bias_fc1', param=nn.Parameter(torch.zeros((n_expert_weights, h))))
        self.register_parameter(name='expert_bias_fc2', param=nn.Parameter(torch.zeros((n_expert_weights, n_output))))

        self.drop1 = nn.Dropout(drop_prob)
        self.drop2 = nn.Dropout(drop_prob)
        self.drop3 = nn.Dropout(drop_prob)

        self.drop_prob = drop_prob

    def initial_alpha(self, shape, rng):
        alpha_bound = np.sqrt(6. / np.prod(shape[-2:]))
        alpha = np.asarray(
            rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape),
            dtype=np.float32)
        return nn.Parameter(torch.from_numpy(alpha))

    def forward(self, x, BC):
        W0, B0, W1, B1, W2, B2 = self.blend(BC)

        x = self.drop1(x)
        x = torch.baddbmm(B0.unsqueeze(2), W0, x.unsqueeze(2))
        x = F.elu(x)
        x = self.drop2(x)
        x = torch.baddbmm(B1.unsqueeze(2), W1, x)
        x = F.elu(x)
        x = self.drop3(x)
        x = torch.baddbmm(B2.unsqueeze(2), W2, x)
        x = x.squeeze(2)
        return x

    def blend(self, BC):
        BC_w = BC.unsqueeze(2).unsqueeze(2)
        BC_b = BC.unsqueeze(2)

        W0 = torch.sum(BC_w * self.expert_weights_fc0.unsqueeze(0), dim=1)
        B0 = torch.sum(BC_b * self.expert_bias_fc0.unsqueeze(0), dim=1)
        W1 = torch.sum(BC_w * self.expert_weights_fc1.unsqueeze(0), dim=1)
        B1 = torch.sum(BC_b * self.expert_bias_fc1.unsqueeze(0), dim=1)
        W2 = torch.sum(BC_w * self.expert_weights_fc2.unsqueeze(0), dim=1)
        B2 = torch.sum(BC_b * self.expert_bias_fc2.unsqueeze(0), dim=1)
        return W0, B0, W1, B1, W2, B2


class GatingNN(nn.Module):
    def __init__(self, n_input, n_expert_weights=6, hidden_dim=512, drop_prob=0.3):
        super(GatingNN, self).__init__()
        self.fc0 = nn.Linear(n_input, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_expert_weights)

        self.drop1 = nn.Dropout(drop_prob)
        self.drop2 = nn.Dropout(drop_prob)
        self.drop3 = nn.Dropout(drop_prob)

        self.drop_prob = drop_prob

    def forward(self, x):
        x = self.drop1(x)
        x = F.elu(self.fc0(x))
        x = self.drop2(x)
        x = F.elu(self.fc1(x))
        x = self.drop3(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x