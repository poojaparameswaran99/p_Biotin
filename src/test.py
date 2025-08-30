import pandas as pd 
import numpy as np 
import torch
from functools import partial
import re
from torch.utils.data import Dataset, DataLoader
import h5py
import ast
from pathlib import Path
import json
import os 
import torch.nn as nn
import hydra 
import sys 
from dataset import LysineDataset
from metrics import compute
from collections import Counter
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from loss import MarginScheduledTripletLossFunction
from hydra.utils import get_original_cwd
sys.path.append(os.path.expanduser('~/soderlinglab/user/pooja/projects/Biotin/src')) ## 
from train import load_data

@hydra.main(version_base=None, config_path="configs", config_name="test")
def main():
    
    return

def test(cfg):
    test_loader = load_data(cfg, 'test')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('training on DEVICE', DEVICE)
    saved_model = cfg['best_model']
    model = hydra.utils.instantiate(cfg['model']['esm2_15b'])
    model_state = torch.load(saved)
    
    return

if __name__ == '__main__':
    main()