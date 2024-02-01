import torch
import torch.nn as nn
import torch.nn.functional  as F
import numpy as np
from decoder import GatingNN, PredictionNet
from encoder import Encoder


class LNSM_Net(nn.Module):
    def __init__(self, output_dim, args, drop_prob=0.3):
        super(LNSM_Net, self).__init__()
        self.rng = np.random.RandomState(1234)
        self.device = torch.device("cuda:0")

        self.num_outputs = output_dim
        self.num_experts = args.num_experts

        self.start_pose = 0
        self.start_goal = args.start_goal

        self.start_gating = args.start_gating
        self.dim_gating = args.dim_gating

        self.gatingNN = GatingNN(self.dim_gating, self.num_experts, 512, drop_prob)
        self.predNN = PredictionNet(args.h_state + args.h_goal, self.num_outputs, self.num_experts, 512, drop_prob)
                
        self.encoder0 = Encoder(self.start_goal, hidden=args.h_state)
        self.encoder1 = Encoder(self.start_gating-self.start_goal, hidden=args.h_goal)

    def forward(self, x):
        BC = self.gatingNN(x[:, -self.dim_gating:])
        out = self.predNN(torch.cat([self.encoder0(x[:, :self.start_goal]),
                                       self.encoder1(x[:, self.start_goal:self.start_gating])], -1), BC)
        return out
