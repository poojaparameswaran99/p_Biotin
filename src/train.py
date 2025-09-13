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
from dataset import TrainDataset, InferenceDataset
from metrics import compute
from collections import Counter
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from loss import MarginScheduledTripletLossFunction
from hydra.utils import get_original_cwd
sys.path.append(os.path.expanduser('~/soderlinglab/user/pooja/projects/Biotin/src')) ## str(Path(get_original_cwd()))

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

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
        for i, batch in enumerate(val_loader):
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
        loss = l/len(val_loader)
        total_labels = torch.cat(total_labels)
        total_preds = torch.cat(total_preds)
        acc, auroc, aupr, CM = compute(total_labels, total_preds, threshold=0.5)
    return acc, auroc, aupr, CM, loss

def load_data(cfg, cross='train'):
    ## implement cross val
    contrastive_dataset_builder = hydra.utils.instantiate(cfg['data'][cross]['dataset'], _partial_=True)
    binary_dataset_builder = hydra.utils.instantiate
    dataset = contrastive_dataset_builder(file=cfg['data']['splits'][cross]['file'])
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
    Loss1 = hydra.utils.instantiate(cfg['Loss1']) ## triplet margin distance loss
    Loss2 = hydra.utils.instantiate(cfg['Loss2']) ## bce
    model = hydra.utils.instantiate(cfg['models']['esm2_15b'])
    BURN_IN = cfg['burn_in_epochs']
    
    # Training parameters
    alternate_epochs = bool(cfg.get('alternate_epochs', False))
    num_epochs = cfg['n_epochs']
    learning_rate = cfg['lr']
    job = cfg['job_type']
    
    #  run name + group
    m   = str(cfg['anneal_fn']['M_0']).replace('.', '')
    time = datetime.now().strftime("%m_%d-%H_%M")
    nae = cfg['n_alternate_epoch']
    NAME = f"alternatEpoch{nae}_{m}" if alternate_epochs else f"bce"
    group = cfg["wandb"]["group"]  
    proj = cfg['project']
    seed = cfg['seed']
    
    ## seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) #

    # start
    min_loss =float('inf')
    max_auroc = float('-inf')
    run = wandb.init(
        entity="soderlinglab-dukecellbio",
        project=proj,
        job_type= job,
        name=NAME,
        group= str(group),
        config=cfg,
        )
    alpha_updates = {0: 'BCE',
                    1: "TML"}
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        if alternate_epochs:
            # alpha*loss1 + (1-alpha)*loss2
            alpha = 1 if (epoch %nae ==0) else 0 ## loss1 gets freq updated
        else:
            alpha = 0
        pnc = Counter()
        for batch_idx, batch in enumerate(train_loader):
            embed, labels= batch # seq_embedding, biotin_pos

            ## count pos and neg
            pnc.update(labels.view(-1).tolist())

            ## move to device
            embed = embed.to(DEVICE)
            labels = labels.to(DEVICE).float()
            pos_len = pos_len.to(DEVICE)

            # run model for this sequence (batch size always 1)
            embed, preds = model(embed)
            embed = embed.squeeze(0) # remove batch dim for inutitive passing thru loss
            labels = labels.float().unsqueeze(-1)  # add columnar dimension to targets

            # Zero gradients
            optimizer.zero_grad(set_to_none=True)

            # Calculate loss
            loss1 = Loss1(embed, pos_len)
            loss2 =  Loss2(preds, labels) ## params

            ## alternate loss backprop w set alpha
            l = alpha*loss1 + (1-alpha)*loss2
#             l = loss2
            l.backward()

            ## optimizer step
            optimizer.step()
            ## update margin
            Loss1.step()

            ## flatten for metrics
            labels = labels.flatten().detach().cpu().clone()
            preds = preds.flatten().detach().cpu().clone()

        ## validate
        v_acc, v_auroc, v_aupr, v_CM, v_loss = validate(cfg, val_loader, model, Loss1, Loss2, alpha)

        ## log val metrics
        val_metrics = {"acc": float(v_acc), "loss": float(v_loss), 'auroc': float(v_auroc), 'aupr': float(v_aupr)}
        run.log(val_metrics)
        print(f'Epoch [{epoch+1}/{num_epochs}]; Val Loss: {v_loss:.4f}, updating {alpha_updates[alpha]}')
        
        ## log best model
        if (v_loss < min_loss or v_auroc > max_auroc) and epoch > BURN_IN:
            safe_pnc = float(pnc) if hasattr(pnc, "__float__") else pnc
            val_metrics.update({'epoch' : epoch, 'pnc': safe_pnc})
            save_path = Path(f'/cwork/pkp14/Biotin/models/{proj}/{str(group)}/{NAME}_{epoch}.pth')
            save_path.parent.mkdir(exist_ok=True, parents=True)   # create parent folder(s) if missing
            opt_path = save_path.with_name(save_path.stem + "_opt.pth")
            extra_data = {'embedding': tuple(embed.shape), 'preds': tuple(preds.shape),
                         'preds_min': preds.min().item(),
                         'preds_max': preds.max().item(), 'margin': float(Loss1.margin),
                          'epoch': epoch, 'last_loss_update': alpha_updates[alpha],
                          'model_path': str(save_path)
                         }
            art = wandb.Artifact(name=f"{NAME}_{epoch}", type="eval", metadata=extra_data)
            with art.new_file("val_metrics.json", mode="w") as f:
                json.dump(val_metrics
                          , f, indent=2)
            with art.new_file("extra_data.json", mode="w") as f:
                json.dump(extra_data, f, indent=2)

            alldata = val_metrics | extra_data
            with open(save_path.parent / f'{save_path.stem}.json', mode='w') as f:
                json.dump(alldata, f, indent=4)
            run.log_artifact(art, aliases=["best"])
            torch.save(model.state_dict(),save_path)
            torch.save(optimizer.state_dict(), opt_path)
            print(f"New best model @ epoch {epoch} saved with loss: {v_loss:.4f}, last loss update {alpha_updates[alpha]},"
                  f"max_auroc: {max_auroc}")
            min_loss = v_loss
            max_auroc = v_auroc
    run.finish()
    return


if __name__ == "__main__":
    main()
