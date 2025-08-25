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
from model import Model
from loss import MyTripletLoss
from metrics import compute
import wandb
from dataset import LysineDataset
import hydra
from omegaconf import DictConfig, OmegaConf
sys.path.append(os.path.expanduser('~/soderlinglab/utils'))

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

## architecture
## extract Lysine residues and parse through model.
### update margin as you train.. with some annealer?
## switch off epochs for diff losses, or...
## weight on each loss. in the addendum or bakcprop.
### how much to update weights on losses 

## any manipulation on BCE loss?9
### update margin as you train.. with some annealer?
## switch off epochs for diff losses, or...
## weight on each loss. in the addendum or bakcprop.
### how much to update weights on losses 


#### actionables
## update lr and margin according to an annealing scheme.
@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig): # 
    cfg = OmegaConf.to_container(cfg, resolve = True)
    
    return

def validate(df, model, TMLcriterion, BCEcriterion, lmbd1, lmbd2, embeddings_path):
    dataset = LysineDataset(df, 'Accession','BiotinPosition','seq', embeddings_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('validation Device ', DEVICE)
    loss = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            embed, labels, pos_len = batch
            embed = embed.to(DEVICE)
            labels = labels.to(DEVICE)
            pos_len = pos_len.to(DEVICE)
            embed, preds = model(embed)
            embed = embed.squeeze(0)
            labels = labels.flatten().cpu().clone()
            preds = preds.flatten().cpu().clone()
            acc, auroc, aupr, CM = compute(labels, preds, threshold=0.5)
            loss1 = TMLcriterion(embed, pos_len)
            loss2 = BCEcriterion(preds, labels) ## params
            # Backward pass
            loss += ((lmbd1*loss1) + (lmbd2*loss2)).item()
    loss = loss/len(dataloader)
    return acc, auroc, aupr, CM, loss

def load_data(cfg, cross='train'):
    ## implement cross val
    dataset_builder = hydra.utils.instantiate(cfg['data']['dataset'], _partial_=True)
    dataset = dataset_builder(file=cfg['data']['dataset'][cross]['file'])
    dataloader_builder = hydra.utils.instantiate(cfg.dataloader, _partial_=True)
    dataloader = dataloader_builder(dataset)
    return dataloader

def train_model(lmbd1=0.5, lmbd2=0.5, alternate_epochs=False):
    # Setup data
    embeddings_path = '/cwork/pkp14/Biotin/esm2_15b_embeds'  # your embeddings path
    train_csv = pd.read_csv(f'~/soderlinglab/user/pooja/projects/Biotin/data/train-val-test/train.csv')
    train_df = train_csv.drop(columns=list(filter(lambda x: x.startswith('Unnamed'), train_csv.columns)))
    val_file = f'~/soderlinglab/user/pooja/projects/Biotin/data/train-val-test/val.csv'
    val_df = pd.read_csv(val_file)
    val_df = val_df.drop(columns=list(filter(lambda x: x.startswith('Unnamed'), val_df.columns)))
    # Create dataset and dataloader
    dataset = LysineDataset(train_df, 'Accession','BiotinPosition','seq', embeddings_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # batch_size=1 for now
    print('train data', train_df)
    print('validation data', val_df)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('training in DEVICE', DEVICE)
    # Training parameters
    num_epochs = 100
    learning_rate = 0.001
    if not alternate_epochs:
        l1n = str(lmbd1).replace('.', '')
        l2n = str(lmbd2).replace('.', '')
        NAME=f'v1_lmbd{l1n}lmbd{l2n}'
    else:
        NAME=f'v1_alternateEpochs'
    min_loss =float('inf')
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="soderlinglab-dukecellbio",
        # Set the wandb project where this run will be logged.
        project="biotin2",
        name=NAME,
        # Track hyperparameters and run metadata.
        config={
                "learning_rate": 0.02,
                "architecture": "MLP",
                "dataset": "train",
                "epochs": num_epochs,
            },
        )
    # Loss function
    TMLcriterion = MyTripletLoss()
    BCEcriterion = nn.BCELoss()
    model = Model()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            embed, labels, pos_len= batch # seq_embedding, biotin_pos
            embed = embed.to(DEVICE)
            labels = labels.to(DEVICE)
            pos_len = pos_len.to(DEVICE)
            # Create model for this sequence
            embed, preds = model(embed)
            embed = embed.squeeze(0)
            # Create optimizer for this model
            # Zero gradients
            optimizer.zero_grad()
            
            # Calculate loss
            loss1 = TMLcriterion(embed, pos_len)
            loss2 = BCEcriterion(preds, labels) ## params
            # Backward pass
            l = ((lmbd1*loss1) + (lmbd2*loss2))
            l.backward()
            optimizer.step()
            ## dim as [b, nr, 1], FLATTEN TO nr only for metrics
            ## flatten for metrics
            labels = labels.flatten().detach().cpu().clone()
            preds = preds.flatten().detach().cpu().clone()
        v_acc, v_auroc, v_aupr, v_CM, v_loss = validate(val_df, model, TMLcriterion, BCEcriterion,
                                                       lmbd1, lmbd2, embeddings_path)
        val_metrics = {"acc": float(v_acc), "loss": float(v_loss), 'auroc': float(v_auroc), 'aupr': float(v_aupr)}
        run.log(val_metrics)
        print(f'Epoch [{epoch+1}/{num_epochs}]; Val Loss: {v_loss:.4f}')
        if v_loss < min_loss and epoch > 10:
            with open(f'/cwork/pkp14/Biotin/models/{NAME}.json','w') as f:
                json.dump(val_metrics, f, indent=4)
            torch.save(model.state_dict(), f'/cwork/pkp14/Biotin/models/{NAME}.pth')
            print(f"New best model saved with loss: {v_loss:.4f}")
            min_loss = v_loss
    run.finish()
    return


if __name__ == "__main__":
    train_model(lmbd1=0.5, lmbd2=0.5)
