import time

import torch 
from torch import nn
import numpy as np
import torch.nn.functional as F
from collections import deque
import random
import gymnasium as gym
import gymnasium_stag_hunt
import matplotlib.pyplot as plt
from tqdm import tqdm

# I will implement a Dueling DQN, where I combine two branches and then I will merge them
#           ┌─ Dense → Dense → V(s)    [1 output]  ─┐
# input ───┤                                          ├→ Q(s,a)
#           └─ Dense → Dense → A(s,a)  [n output] ──┘

# The first branch estimates V(s), i.e. the value function
# The second branch estimates A(s,a), i.e. the advantage function
# The final estimate of Q(s,a) is then obtained as Q(s,a) = V(s) + A(s,a) - mean(A(s,a))

class DQN(nn.Module):
    def __init__(self, state_size, action_size, num_nodes=64):
        super().__init__()
        # stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(state_size, num_nodes),
            nn.ReLU(),
            nn.Linear(num_nodes, num_nodes),
            nn.ReLU(),
            nn.Linear(num_nodes, 1)  
        )

        # stream A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(state_size, num_nodes),
            nn.ReLU(),
            nn.Linear(num_nodes, num_nodes),
            nn.ReLU(),
            nn.Linear(num_nodes, action_size)
        )
        
    def forward(self, x):
        v = self.value_stream(x)        # shape (batch, 1)
        a = self.advantage_stream(x)    # shape (batch, action_size)

        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        return v + a - a.mean(dim=-1, keepdim=True)

