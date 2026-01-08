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

@torch.no_grad()
def test(cfg):
    ## load data
    test_loader = load_data(cfg, 'test')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    saved_model_folder = cfg['best_model']
    saved_model_path = Path(saved_model_folder) / 'best_model.pth'
    model = hydra.utils.instantiate(cfg['models']['esm2_15b'])
    model_state = torch.load(saved_model_path, weights_only=True)
    model.load_state_dict(model_state)
    model.to(DEVICE)
    model.eval()
    pnc = Counter()
    test_binary_dataloader, _ = load_data(cfg, cross='test')
    PROJECT = cfg.get('project', str(saved_model_path).replace('/', '_'))
    GROUP = Path(saved_model_path).parent.stem
    RUN_ID = cfg.get('wandb_run_id', str(saved_model_path).replace('/', '_'))  # or set to your known id like "moidmqea"
    NAME = Path(saved_model_path).parent.stem
    RUN = wandb.init(
        entity="soderlinglab-dukecellbio",
        project=PROJECT,
        id=RUN_ID,           # None => new run; or set to your known id to resume
        job_type="testing",
        name= Path(saved_model_path).parent.stem if RUN_ID else None,
        group=GROUP,
        config=cfg
    )
    labels = []
    preds = []

    for i, batch in enumerate(test_binary_dataloader):
        e, l = batch
        e, l = e.to(DEVICE), l.to(DEVICE, dtype=torch.float32)
        pnc.update(l.tolist())
        _ , p = model(e)
        p = torch.sigmoid(p)
        labels.append(l.flatten().cpu().clone())
        preds.append(p.flatten().cpu().clone())
            
    labels = torch.cat(labels, dim=0).cpu()
    preds  = torch.cat(preds,  dim=0).cpu()
    df = pd.DataFrame({'labels': labels.numpy(), 'preds':preds.numpy()})
    safe_pnc = dict(pnc) if hasattr(pnc, "items") else (float(pnc) if hasattr(pnc, "__float__") else pnc)
    
    ## get test metrics
    acc, auroc, aupr, CM = compute(labels, preds, threshold=0.5)
    test_metrics = {'model_path': NAME,
                    'test_accuracy': float(acc),
                   'test_auroc': float(auroc),
                   'test_aupr': float(aupr), ## cm
                   'pnc': safe_pnc}
    save_test_path = Path(f'/cwork/pkp14/Biotin/models/{PROJECT}/{NAME}')
    save_test_path.mkdir(exist_ok=True, parents=True)
    with open(save_test_path / 'test_metrics.json', mode='w') as f:
        json.dump(test_metrics, f ,indent=4)
    df.to_csv(save_test_path / 'test_preds.csv', index=False)
    
    print(f'test metrics written to {save_test_path}')
    art = wandb.Artifact(name=f"{NAME}_eval", type="eval", metadata=test_metrics)
    art.add_file(str(save_test_path / 'test_metrics.json'))
    art.add_file(str(save_test_path / 'test_preds.csv'))
    wandb.log_artifact(art)
    logged = RUN.log_artifact(art)
    logged.wait()                 # ensure artifact upload completes
    print("Run URL:", RUN.url)    # sa
    wandb.finish()                # flush everything
    return

if __name__ == '__main__':
    main()
