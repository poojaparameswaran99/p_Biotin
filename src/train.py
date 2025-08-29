import pandas as pd 
import numpy as np 
import torch
from functools import partial
import math
import re
from torch.utils.data import Dataset, DataLoader
import h5py
import ast
from pathlib import Path
from datetime import datetime
import json
import os 
import torch.nn as nn
import hydra 
import random
import sys
sys.path.append(os.path.expanduser('~/soderlinglab/user/pooja/projects/Biotin/src'))
from dataset import LysineDataset
from metrics import compute  # whatever you need
from collections import Counter
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from loss import MarginScheduledTripletLossFunction
from hydra.utils import get_original_cwd
sys.path.append(os.path.expanduser('~/soderlinglab/user/pooja/projects/Biotin/src')) ## str(Path(get_original_cwd()))

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
@hydra.main(version_base=None, config_path="configs", config_name="trainn")
def main(cfg: DictConfig): # 
    cfg = OmegaConf.to_container(cfg, resolve = True)
    train_model(cfg)
    return


def validate(cfg, val_loader, model, Loss1, Loss2, alpha):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    l = 0
    model.eval()
    total_labels = []
    total_preds = []
    with torch.no_grad():
        for batch in val_loader:
            embed, labels, pos_len = batch
            embed = embed.to(DEVICE)
            labels = labels.to(DEVICE)
            pos_len = pos_len.to(DEVICE)
            embed, preds = model(embed)
            embed = embed.squeeze(0)
            preds = preds.squeeze(-1)
            total_labels.append(labels.flatten().cpu().clone())
            total_preds.append(preds.flatten().cpu().clone())
            loss1 = Loss1(embed, pos_len)
            loss2 = Loss2(preds, labels) ## params
            # Backward pass
            l += (alpha*loss1 + (1-alpha)*loss2).item()
#             l += loss2
    loss = l/len(val_loader)
    total_labels = torch.cat(total_labels)
    total_preds = torch.cat(total_preds)
    acc, auroc, aupr, CM = compute(total_labels, total_preds, threshold=0.5)
    return acc, auroc, aupr, CM, loss

def load_data(cfg, cross='train'):
    ## implement cross val
    dataset_builder = hydra.utils.instantiate(cfg['data']['dataset'], _partial_=True)
    dataset = dataset_builder(file=cfg['data']['splits'][cross]['file'])
    dataloader_builder = hydra.utils.instantiate(cfg['data']['dataloader'], _partial_=True)
    dataloader = dataloader_builder(dataset)
    return dataloader

def train_model(cfg):
    ## data
    train_loader = load_data(cfg, 'train')
    val_loader = load_data(cfg, cross='val')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('training on DEVICE', DEVICE)
    
    ## instantiate
    Loss1 = hydra.utils.instantiate(cfg['Loss1'])
    Loss2 = hydra.utils.instantiate(cfg['Loss2'])
    model = hydra.utils.instantiate(cfg['models']['esm2_15b'])
    BURN_IN = cfg['burn_in_epochs']
    # Training parameters
    alternate_epochs = bool(cfg.get('alternate_epochs', False))
    num_epochs = cfg['n_epochs']
    learning_rate = cfg['lr']
    job = cfg['job_type']
    
    # Nice run name + group
    l1n = str(cfg['loss1_weight']).replace('.', '')
    l2n = str(1 - cfg['loss1_weight']).replace('.', '')
    m   = str(cfg['anneal_fn']['M_0']).replace('.', '')
    time = datetime.now().strftime("%m_%d-%H_%M")
    nae = cfg['n_alternate_epoch']
    NAME = f"alternatEpoch{nae}_{m}" if alternate_epochs else f"not"
    group = cfg["wandb"]["group"]   # <- this will be "euclidean"
    proj = cfg['project']
    seed = cfg['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # start
    min_loss =float('inf')
    run = wandb.init(
        entity="soderlinglab-dukecellbio",
        project=proj,
        job_type= job,
        name=NAME,
        group=str(group),
        config=cfg,
        )
    model.to(DEVICE)
    ## make hydra compatible
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        if alternate_epochs:
            alpha = 1 if (epoch %nae ==0) else 0 ## update bce every 5 epochs
        else:
            alpha = cfg['loss1_weight']
        pnc = Counter()
        for batch_idx, batch in enumerate(train_loader):
            embed, labels, pos_len= batch # seq_embedding, biotin_pos
            pnc.update(labels.view(-1).tolist())
            embed = embed.to(DEVICE)
            labels = labels.to(DEVICE).float()
            pos_len = pos_len.to(DEVICE)
            # Create model for this sequence
            embed, preds = model(embed)
            embed = embed.squeeze(0)
            # Create optimizer for this model
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)
            preds = preds.squeeze(-1)
            # Calculate loss
            loss1 = Loss1(embed, pos_len)
            loss2 =  Loss2(preds, labels) ## params
            l = alpha*loss1 + (1-alpha)*loss2
#             l = loss2
            l.backward()
            ## optimizer step
            optimizer.step()
            ## update margin
            Loss1.step()
            ## dim as [b, nr, 1], FLATTEN TO nr only for metrics
            ## flatten for metrics
            labels = labels.flatten().detach().cpu().clone()
            preds = preds.flatten().detach().cpu().clone()
        v_acc, v_auroc, v_aupr, v_CM, v_loss = validate(cfg, val_loader, model, Loss1, Loss2, alpha)
        val_metrics = {"acc": float(v_acc), "loss": float(v_loss), 'auroc': float(v_auroc), 'aupr': float(v_aupr)}
        run.log(val_metrics)
        print(f'Epoch [{epoch+1}/{num_epochs}]; Val Loss: {v_loss:.4f}')
        if v_loss < min_loss and epoch > BURN_IN:
            safe_pnc = float(pnc) if hasattr(pnc, "__float__") else pnc
            val_metrics.update({'epoch' : epoch, 'pnc': safe_pnc})
            extra_data = {'embedding': tuple(embed.shape), 'preds': tuple(preds.shape),
                         'preds_min': preds.min().item(),
                         'preds_max': preds.max().item(), 'margin': float(Loss1.margin)
                         }
            
            art = wandb.Artifact(name=f"{NAME}-eval", type="eval", metadata=extra_data)
            with art.new_file("val_metrics.json", mode="w") as f:
                json.dump(val_metrics
                          , f, indent=2)
            with art.new_file("extra_data.json", mode="w") as f:
                json.dump(extra_data, f, indent=2)
            run.log_artifact(art, aliases=["best"])
            torch.save(model.state_dict(), f'/cwork/pkp14/Biotin/models/{NAME}.pth')
            print(f"New best model @ epoch {epoch} saved with loss: {v_loss:.4f}")
            print(f'POS:NEG ratio {pnc}')
            min_loss = v_loss
    run.finish()
    return


if __name__ == "__main__":
    main()

