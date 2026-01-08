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
from dataset import BinaryDataset, ContrastiveDataset
from metrics import compute
from collections import Counter
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from loss import MarginScheduledTripletLossFunction
from hydra.utils import get_original_cwd
sys.path.append(os.path.expanduser('~/soderlinglab/user/pooja/projects/Biotin/src')) ## str(Path(get_original_cwd()))

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig): 
    ## parse hydra config to dictionary
    cfg = OmegaConf.to_container(cfg, resolve = True)
    
    ## train model
    train_model(cfg)
    return

@torch.no_grad()
def validate(val_loader, model, optimizer, bcelogits_loss, loss_used, min_loss, max_auroc,
             epoch, PROJECT, GROUP, NAME, RUN, BURN_IN, DEVICE):
    ## initialize
    v_loss = 0
    
    ## model in eval mode
    model.eval()
    labels = []
    preds = []
    pnc = Counter()
    for i, batch in enumerate(val_loader):
        e, l = batch
        e, l= e.to(DEVICE), l.to(DEVICE, dtype=torch.float32)
        pnc.update(l.tolist())
        _, p_logits = model(e)
        v_loss += bcelogits_loss(p_logits, l).item()
        p = torch.sigmoid(p_logits)
        labels.append(l.flatten().cpu().clone())
        preds.append(p.flatten().cpu().clone())
    v_loss = v_loss / len(labels)
    ## is it necessary to torch cat labels and preds
    ## convert preds to logits 
    preds  = torch.cat(preds,  dim=0).cpu()
    labels = torch.cat(labels, dim=0).cpu()
    acc, auroc, aupr, CM, recall, precision, barplot, heatmap = compute(labels, preds, threshold=0.5)
    ## log val metrics
    val_metrics = {"Validation/Accuracy": float(acc),
                   "Validation/Loss": float(v_loss), 
                   'Validation/AUROC': float(auroc), 
                   'Validation/AUPR': float(aupr), 
                   'Validation/Precision': float(precision),
                   'Validation/Recall': float(recall),
                    'epoch': epoch}
    RUN.log(val_metrics, step=epoch, commit=True)
    
    min_loss, max_auroc = save_best_model(model, optimizer, v_loss, min_loss, auroc,
                                          max_auroc, epoch, BURN_IN, PROJECT, GROUP,
                                          NAME, run=RUN, loss_used=loss_used, pnc=pnc)
    return min_loss, max_auroc

def load_data(cfg, cross='train'):
    esm_engine = hydra.utils.instantiate(cfg.embedding)
    ## binary
    binary_builder = hydra.utils.instantiate(cfg['data']['binary']['dataset'])
    binary_dataset = binary_builder(file=cfg['data']['binary_dataset_files'][cross]['file'],
                                   esm_engine=esm_engine)
    binary_loader = hydra.utils.instantiate(cfg['data']['binary']['dataloader'],
                                            dataset=binary_dataset)
    # contrastive
    contrastive_dataset = hydra.utils.instantiate(cfg['data']['contrastive']['dataset'])
    contrastive_loader =  hydra.utils.instantiate(cfg['data']['contrastive']
                                                  ['dataloader'], dataset=contrastive_dataset)
    return binary_loader, contrastive_loader

def run_train_binary_loader(binary_loader, val_loader, model, 
                            optimizer, bcelogits_loss, min_loss, 
                            max_auroc, n_epoch, PROJECT, GROUP, NAME, BURN_IN, RUN, DEVICE):
    pnc = Counter()
    model.train()
    labels = []
    preds = []
    e_loss = 0
    for bidx, batch in enumerate(binary_loader):
        optimizer.zero_grad() ## prevent accumulation across training steps
        e, l = batch
        e, l = e.to(DEVICE), l.to(DEVICE, dtype=torch.float32)
        pnc.update(l.view(-1).tolist())
        _, p = model(e)
        loss = bcelogits_loss(p, l)
        loss.backward()
        optimizer.step()
        labels.append(l.flatten().detach().cpu().clone())
        preds.append(p.flatten().detach().cpu().clone())
        e_loss += loss.item()
    e_loss = e_loss / len(labels)
    pnc = float(pnc) if hasattr(pnc, "__float__") else pnc
    labels = torch.cat(labels, dim=0).cpu()
    preds  = torch.cat(preds,  dim=0).cpu()
    
    acc, auroc, aupr, CM, recall, precision, barplot, heatmap = compute(labels, preds, threshold=0.5)
    RUN.log({'Train/Loss': e_loss, 
             'epoch': n_epoch,
             'Train/Accuracy': acc,
             'Train/AUROC': auroc, 
             'Train/AUPR': aupr,
            'Train/Recall': recall,
             'Train/Precision': precision},
            step=n_epoch)
    
    min_loss, max_auroc = validate(val_loader, model, optimizer,
                                   bcelogits_loss, 'BCElogits', min_loss, 
                                   max_auroc, n_epoch, PROJECT, GROUP, NAME, RUN, BURN_IN, DEVICE)
    
    return model, optimizer, min_loss, max_auroc

def run_train_contrastive_loader(contrastive_loader, val_loader, model, 
                                 optimizer, contrastive_loss, bcelogits_loss, 
                                 min_loss, max_auroc, n_epoch, PROJECT, 
                                 GROUP, NAME, BURN_IN, RUN, DEVICE):
    model.train()
    labels = []
    preds = []
    e_loss = 0
    for bidx, batch in enumerate(contrastive_loader):
        optimizer.zero_grad() ## prevent accumulation across training steps
        e = batch
        e = e.to(DEVICE)
        o, _ = model(e) 
        loss = contrastive_loss(o)
        loss.backward()
        optimizer.step()
        contrastive_loss.step()
        e_loss += loss.item()

    RUN.log({'Train/Loss': e_loss, 'epoch': n_epoch}, step=n_epoch)
    min_loss, max_auroc = validate(val_loader, model, 
                                   optimizer, bcelogits_loss , 
                                   'Contrastive_Loss', min_loss, 
                                   max_auroc, n_epoch, PROJECT, GROUP,
                                   NAME, RUN, BURN_IN, DEVICE)
    return model, optimizer, min_loss, max_auroc

def train_model(cfg):
    # changing params
    margin   = str(cfg['anneal_fn']['M_0']).replace('.', '')
    time = datetime.now().strftime("%m_%d-%H_%M")
    
    # Training parameters
    BURN_IN = cfg['burn_in_epochs']
    ALTERNATE_EPOCHS = bool(cfg.get('alternate_epochs', False))
    N_EPOCHS = cfg['n_epochs']
    LR = cfg['lr']
    JOBTYPE = cfg['job_type']
    NAE = cfg['n_alternate_epoch']
    NAME = f"alternatEpoch{NAE}_{margin}" if ALTERNATE_EPOCHS else f"bce"
    GROUP = cfg["wandb"]["group"]  
    PROJECT = cfg['project']
    SEED = cfg['seed']

    ## set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED) #

    # initialize model check params
    min_loss =float('inf')
    max_auroc = float('-inf')    

    ## initialize wandb
    RUN = wandb.init(
        entity="soderlinglab-dukecellbio",
        project=PROJECT,
        job_type= JOBTYPE,
        name=NAME,
        group= str(GROUP),
        config=cfg,
        )
    
    wandb.define_metric("epoch")
    wandb.define_metric(name = "Train/*", step_metric="epoch")
    wandb.define_metric(name= "Validation/*", step_metric="epoch")
    

    ## data
    train_binary_dataloader, train_contrastive_dataloader= load_data(cfg, 'train')
    val_binary_dataloader, _ = load_data(cfg, cross='validation')
    
    ## instantiate working functions
    contrastive_loss = hydra.utils.instantiate(cfg['Loss1']) ## triplet margin distance loss
    bcelogits_loss = hydra.utils.instantiate(cfg['Loss2']) ## bce
    model = hydra.utils.instantiate(cfg['models']['esm2_15b'])
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    alpha_updates = {0: bcelogits_loss,
                     1: contrastive_loss}
    
    for e in range(N_EPOCHS):
        if e % NAE == 0:
            model, optimizer, min_loss, max_auroc = run_train_binary_loader(train_binary_dataloader, 
                                                                            val_binary_dataloader, model, 
                                                                            optimizer, bcelogits_loss,
                                                                            min_loss, max_auroc, e, PROJECT,
                                                                            GROUP, NAME, BURN_IN, RUN, DEVICE)
        else:
            model, optimizer, min_loss, max_auroc = run_train_contrastive_loader(train_contrastive_dataloader, val_binary_dataloader, model, optimizer,
                                                                                 contrastive_loss,
                                                                                 bcelogits_loss, min_loss, 
                                                                                 max_auroc, e, PROJECT, GROUP,
                                                                                 NAME, BURN_IN, RUN, DEVICE)
    RUN.finish()
    return



def save_best_model(model, optimizer, val_loss, min_loss, val_auroc, max_auroc, epoch, burn_in, project,
                    group, name, run, loss_used, pnc=None):
    improved = (val_loss < min_loss) or (val_auroc > max_auroc)
    if not (improved and epoch > burn_in):
        return min_loss, max_auroc

    save_dir = Path(f"/cwork/pkp14/Biotin/models/{project}/{group}/{name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"best_model.pth"
    opt_path  = save_dir / f"best_optimizer.pth"
    
    # serialize-safe
    if pnc:
        safe_pnc = dict(pnc) if hasattr(pnc, "items") else (float(pnc) if hasattr(pnc, "__float__") else pnc)

    val_metrics = {
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "val_auroc": float(val_auroc),
        "posneg_counter": safe_pnc,
    }

    extra_data = {
        "loss_used": str(loss_used) if loss_used is not None else None,
        "epoch": int(epoch),
        "model_path": str(save_path),
    }

    # sidecar JSON
    with open(save_dir / f"metrics.json", "w") as f:
        json.dump({**val_metrics, **extra_data}, f, indent=2)

    # weights
    torch.save(model.state_dict(), save_path)
    torch.save(optimizer.state_dict(), opt_path)

    art = wandb.Artifact(name=f"metrics", type="eval", metadata=extra_data)
    with art.new_file("val_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)
    with art.new_file("extra_data.json", "w") as f:
        json.dump(extra_data, f, indent=2)
    run.log_artifact(art, aliases=["best"])

    print(f"New best @ epoch {epoch}: loss={val_loss:.4f}, auroc={val_auroc:.4f}")
    return float(val_loss), float(val_auroc)


if __name__ == "__main__":
    main()
