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


class Model6(nn.Module):
    def __init__(self, in_dim=5120, hidden_dim1=3415, hidden_dim2=2560, out_dim=1280):
        super(Model6, self).__init__()
        self.indim = in_dim
        self.hiddim1 = hidden_dim1
        self.hiddim2 = hidden_dim2
        self.outdim = out_dim
        
        self.fc1 = nn.Linear(in_dim , hidden_dim1)
        self.ln1 = nn.LayerNorm(hidden_dim1)
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.ln2 = nn.LayerNorm(hidden_dim2)

        self.fc3 = nn.Linear(hidden_dim2, out_dim)
        self.ln3 = nn.LayerNorm(out_dim)
        
        self.fc4 = nn.Linear(out_dim, 1)

        
        self.Sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)
        
        self.block1 = nn.Sequential(self.fc1, self.ln1)
        self.block2 = nn.Sequential(self.fc2, self.ln2,  nn.ReLU()) # embed
        self.block3 = nn.Sequential(self.fc3, self.ln3, nn.ReLU())
        self.block4 = nn.Sequential(self.fc4)

        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x0 = self.block3(x)
        x1 = self.block4(x0)
        return x0, x1
    

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
        x = self.Sigmoid(x)
        return sigmoidal_proj, x 


## Model2 == mlp + sigmoid
class Model3(nn.Module):
    def __init__(self, in_dim=5120,hidden_dim=2560, out_dim=1280):
        super(Model3, self).__init__()
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
        x0 = self.fc3(x)
        x1 = self.Sigmoid(x0)
        return x0, x1

class Model4(nn.Module):
    def __init__(self, in_dim=5120,hidden_dim=2560, out_dim=1280):
        super(Model4, self).__init__()
        self.indim = in_dim
        self.hiddim = hidden_dim
        self.outdim = out_dim
        
        self.fc1 = nn.Linear(in_dim , hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)

        self.fc3 = nn.Linear(out_dim, 1)
        
        self.Sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        
        self.block1 = nn.Sequential(self.fc1, self.ln1, nn.ReLU())
        self.block2 = nn.Sequential(self.fc2, self.ln2) # embed
        self.block3 = nn.Sequential(nn.ReLU(), self.fc3, self.Sigmoid)

        
    def forward(self, x):
        x = self.block1(x)
        x0 = self.block2(x)
        x1 = self.block3(x0)
        return x0, x1

class Model5(nn.Module):
    def __init__(self, in_dim=5120,hidden_dim=2560, out_dim=1280):
        super(Model5, self).__init__()
        self.indim = in_dim
        self.hiddim = hidden_dim
        self.outdim = out_dim
        
        self.fc1 = nn.Linear(in_dim , hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)

        self.fc3 = nn.Linear(out_dim, 1)
        
        self.Sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        
        self.block1 = nn.Sequential(self.fc1, self.ln1, nn.ReLU())
        self.block2 = nn.Sequential(self.fc2, self.ln2) # embed
        self.block3 = nn.Sequential(nn.ReLU(), self.fc3)

        
    def forward(self, x):
        x = self.block1(x)
        x0 = self.block2(x)
        x1 = self.block3(x0)
        return x0, x1
    
    