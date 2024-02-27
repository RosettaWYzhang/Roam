# this file contains frame encoder and goal encoder
import os
from torch import nn


class Encoder(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, input_dim, hidden=512, drop_prob=0.3):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Dropout(drop_prob),
      nn.Linear(input_dim, hidden),
      nn.ELU(),
      nn.Dropout(drop_prob),
      nn.Linear(hidden, hidden),
      nn.ELU()
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)
