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

## things to do: mainstream dataloader for training
## for test take al lysines, know the lysines that are positive and run metric analysis.
@hydra.main(version_base=None, config_path="configs", config_name="test")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve = True)
    test(cfg)
    return


def test(cfg):
    ## load data
    test_loader = load_data(cfg, 'test')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('training on DEVICE', DEVICE)
    saved_model_path = cfg['best_model']
    model = hydra.utils.instantiate(cfg['models']['esm2_15b'])
    model_state = torch.load(saved_model_path, weights_only=True)
    model.load_state_dict(model_state)
    model.to(DEVICE)
    model.eval()
    pnc = Counter()
    total_labels = []
    total_preds = []

    PROJECT = cfg.get('project', 'default-project')
    GROUP = Path(saved_model_path).parent.stem
    RUN_ID = cfg.get('wandb_run_id', None)  # or set to your known id like "moidmqea"
    NAME = Path(saved_model_path).stem
    run = wandb.init(
        entity="soderlinglab-dukecellbio",
        project=PROJECT,
        id=RUN_ID,           # None => new run; or set to your known id to resume
        resume="must" if RUN_ID else None,
        job_type="testing",
        name= Path(saved_model_path).stem if RUN_ID else None,
        group=GROUP,
        config=cfg
    )

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            embed, labels, pos_len = batch
            embed = embed.to(DEVICE)
            labels = labels.to(DEVICE)
            pos_len = pos_len.to(DEVICE)
            
            pnc.update(labels.view(-1).tolist())
            
            embed, preds = model(embed)
            embed = embed.squeeze(0)
            preds = preds.squeeze(-1)
            
            total_labels.append(labels.flatten().cpu().clone())
            total_preds.append(preds.flatten().cpu().clone())
            
    total_labels = torch.cat(total_labels)
    total_preds = torch.cat(total_preds)

    ## get test metrics
    acc, auroc, aupr, CM = compute(total_labels, total_preds, threshold=0.5)
    test_metrics = {'acc': float(acc), 'auroc':float(auroc), 'aupr': float(aupr), 'CM': np.asarray(CM).tolist()}

    safe_pnc = float(pnc) if hasattr(pnc, "__float__") else pnc
    test_metrics.update({'model_name': saved_model_path,'pnc': safe_pnc})

    # Log scalars
    wandb.log({
        "test/acc": test_metrics["acc"],
        "test/auroc": test_metrics["auroc"],
        "test/aupr": test_metrics["aupr"]
    })

    # Save & log as artifact
    out_dir = Path(f'/cwork/pkp14/Biotin/models/{GROUP}')
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f'{NAME}_test-results.json'
    with open(save_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)

    art = wandb.Artifact(name=f"{NAME}_{GROUP}_test", type="eval", metadata=test_metrics)
    art.add_file(str(save_path))
    run.log_artifact(art, aliases=["test_results"])

    run.finish()
    return acc, auroc, aupr, CM

if __name__ == '__main__':
    main()
