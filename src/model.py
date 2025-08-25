import functools # partial, reduce?
import pandas as pd
import numpy as np 
import os
import sys
from torch import nn
import torch.nn.functional as F
import torch
import h5py
from pathlib import Path

## Model1 = mlp w scalar sigmoidal
class Model1(nn.Module):
    def __init__(self, in_dim=5120,hidden_dim=2560, out_dim=1280):
        super(Model1, self).__init__()
        self.indim = in_dim
        self.hiddim = hidden_dim
        self.outdim = out_dim
        self.fc1 = nn.Linear(in_dim , hidden_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_normal_(self.fc2.weight)
        self.fc3 = nn.Sequential(nn.ReLU(), nn.Linear(out_dim, 1))
        nn.init.xavier_normal_(self.fc3[1].weight)
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        TMLx = self.fc2(x) ## out 1280
        x = self.fc3(TMLx)
        x = self.Sigmoid(x).squeeze(-1) # batch, rows, cols, [1, RES_NUM, 1]
        return TMLx, x # only removes dim =1, picks last

## Model2 == mlp + sigmoid
class Model2(nn.Module):
    def __init__(self, in_dim=5120,hidden_dim=2560, out_dim=1280):
        super(Model2, self).__init__()
        self.indim = in_dim
        self.hiddim = hidden_dim
        self.outdim = out_dim
        self.fc1 = nn.Linear(in_dim , hidden_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(out_dim, 1)
        nn.init.xavier_normal_(self.fc3.weight)
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) ## out 1280
        sigmoidal_proj = self.Sigmoid(x) ## out 1280
        x = self.fc3(sigmoidal_proj)
        return sigmoidal_proj, x 
