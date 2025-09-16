from functools import partial, reduce
import pandas as pd
import numpy as np 
import os
import sys
from torch import nn
import torch.nn.functional as F
import torch
import h5py
from typing import Optional
from torch import Tensor
from pathlib import Path
from functools import partial
from torch.nn.modules.loss import _WeightedLoss, _Loss
## distance funcs

def sigmoid_cosine_distance_p(x, y, p=1): # p is weighting factor
    sig = torch.nn.Sigmoid()
    cosine_sim = torch.nn.CosineSimilarity()
    return (1 - sig(cosine_sim(x, y))) ** p

DISTANCE_FN_DICT = {'sigmoid_cosine_distance_p': sigmoid_cosine_distance_p,
                   'euclidean': None}

def tanh_decay(M_0, N_restart, x):
    return float(M_0 * (1.0 - np.tanh(2.0 * x / max(1, N_restart))))

def cosine_anneal(M_0, N_restart, x):
    return float(0.5 * M_0 * (1.0 + np.cos(np.pi * x / max(1, N_restart))))

def no_decay(M_0, N_restart, x):
    return float(M_0)

MARGIN_FN_DICT = {
    "tanh_decay": tanh_decay,
    "cosine_anneal": cosine_anneal,
    "no_decay": no_decay,
}

class MarginScheduledTripletLossFunction():
    def __init__(self, distance_fn, anneal_fn, N_restart: int =10, seed=123, step_per_call=False):
        ## could change into sigmoid_cosine_distance_p
        self._dist = distance_fn        
        self._anneal = anneal_fn  
        self.N_restart = int(N_restart)
        self._step = 0
        self.step_per_call = bool(step_per_call)
        self.M_curr = float(self._anneal(x=0))
        self.seed = seed
    
    ## only referenced upon call on local script
    @property
    def margin(self) -> float:
        return float(self.M_curr)

    def step(self):
        self._step += 1
        x = self._step % self.N_restart
        self.M_curr = float(self._anneal(x=x)) ## update margin fn

    def reset(self):
        self._step = 0
        self.M_curr = self._anneal(x=0) ## update_margin_fn


    def __call__(self, embedding: torch.Tensor):
        n_embedding = embedding.size()
        positives = [embedding.select(0, i) for i in n_embedding if i%3 ==0]
        anchors = [embedding.select(0, i) for i in n_embedding if i%3 ==1]
        
        ## test in console...
        positives = embedding.index_select(0, torch.tensor(n_embedding[0::3]))
        anchors   = embedding.index_select(0, torch.tensor(n_embedding[1::3]))
        negatives = embedding.index_select(0, torch.tensor(n_embedding[2::3]))
        positives = embedding[:,  0, :]
        anchors   = embedding[:, 1, :]
        negatives = embedding[:,2,:]
        loss = F.triplet_margin_with_distance_loss(
            anchors, positives, negatives,
            distance_function=self._dist,
            margin=self.margin,
            swap=False,
            reduction='mean'
        )

        if self.step_per_call:
            self.step()        
        return loss
