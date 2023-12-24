import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Defining the model class
class Dummy(nn.Module):

    def __init__(self, n_observation_feats, n_actions, batch_size):
        super(DQfD, self).__init__()
        self.layer1 = nn.Linear(n_observation_feats, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        self.batch_size = batch_size

    # Called with either one element to determine next action, or a batch
    def forward(self, x, for_optimization=True):
        # When using for optimization, a batch of inputs is passed in. In this case, reshape. When using for selecting actions, only one state is 
        # passed. In this case, the shape is already correctly set. Hence no reshaping is needed.
        if for_optimization: 
            x = torch.reshape(x, (self.batch_size,-1))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)