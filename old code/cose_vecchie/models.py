
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
import time 
import numpy as np
import pandas as pd
from numpy.random import choice
from scipy.special import softmax


class DQN_CNN(nn.Module):
    def __init__(self, height, width, channels, out, kernel_size=(2,2), stride=(1,1)):
        super().__init__()
        
        # Model policy is a 3D-CNN with two convolution layers (with a pooling and a batch_normalization)
        pool_kernel, pool_stride = (1,1), (1,1)
        self.out = out
        self.height = height
        self.width = width
        self.channels = channels
        self.device = torch.device("cuda" if torch .cuda.is_available() else "cpu")
        
        self.conv1 = nn.Conv2d(in_channels=self.channels, out_channels=12, kernel_size=kernel_size, stride=stride)
        self.pool1 = nn.MaxPool2d(pool_kernel, stride=pool_stride)
        self.bn1 = nn.BatchNorm2d(12)
        
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=kernel_size, stride=stride)
        self.pool2 = nn.MaxPool2d(pool_kernel, stride=pool_stride)
        self.bn2 = nn.BatchNorm2d(24)

        def conv2d_size_out(size, kernel_size = kernel_size, stride = stride):
            h = (size[0] - (kernel_size[0] - 1)) // stride[0]
            w = (size[1] - (kernel_size[1] - 1)) // stride[1]
            return (h,w)
        
        conv_height,conv_width = conv2d_size_out(conv2d_size_out((self.height,self.width)))

        
        # The first flatten layer is common for every policy but the last two layer are different based on which are the abstract 
        # actions that the policy have to achieve
        self.fc1 = nn.Linear(in_features = 24 * conv_width * conv_height, out_features = 100)
        self.dict_layers = {}


    def forward(self, x, abstract_action):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.pool1(self.conv1(x))))
        x = F.relu(self.bn2(self.pool2(self.conv2(x))))

        t = x.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        
        # After the first flatten layer take the last two layers from the dict layer initialized above 
        fc2, out = self.dict_layers.setdefault(abstract_action, (nn.Linear(in_features = 100, out_features = 25),nn.Linear(in_features = 25, out_features = self.out)))
        t = F.relu(fc2(t))
        t = F.softmax(out(t))
        return t
    
class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def get_current(policy_net, states, actions,abstract_action):
        return policy_net(states,abstract_action).squeeze(1).gather(dim=1, index=actions.unsqueeze(-1))
    
    @staticmethod
    def get_next(target_net, next_states,abstract_action):
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states,abstract_action).squeeze(1).max(dim=1)[0].detach()
        return values
    
from unicodedata import name
import pandas as pd
from scipy.special import softmax

class QTable():
    def __init__(self):
        self.q_table = pd.DataFrame()
    
    def take(self,state,action):
        state=tuple(state)
        if not action:
            action = -1
        try:
            #state exists
            state_df = self.q_table.loc[state]
            try:
                #action exists
                return state_df[action]
            except:
                #action does not exist
                self.q_table[action] = self.q_table.mean(axis=1)
                self.q_table.apply(softmax,axis=1)
                return self.q_table.loc[state,action]
        except:
            #state does not exist
            if not list(self.q_table.columns):
                new_row = pd.Series(np.zeros(self.q_table.shape[1]),name = state)
            else:
                new_row = pd.Series(np.zeros(self.q_table.shape[1]),name = state, index= self.q_table.columns)
            print(new_row)
            self.q_table = self.q_table.append(new_row, ignore_index=False)
            state_df = self.q_table.loc[state]
            try:
                v = state_df[action]
                return v
            except:
                #action does not exist
                if self.q_table.mean(axis=1).isna().any():
                    self.q_table[action] = np.zeros(self.q_table.shape[0])
                else:
                    self.q_table[action] = self.q_table.mean(axis=1)
                self.q_table.apply(softmax,axis=1)
                return self.q_table.loc[state,action]


    def val_max(self,state):
        state = tuple(state)
        return self.q_table.loc[state].max()
        
    def idx_max(self,state):
        state = tuple(state)
        return self.q_table.loc[state].idxmax(axis = "columns")
        
    
    
        