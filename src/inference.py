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
from loss import MarginScheduledTripletLossFunction
from hydra.utils import get_original_cwd
from model import Model6
sys.path.append(os.path.expanduser('~/soderlinglab/user/pooja/projects/Biotin/src')) ## 
sys.path.append(os.path.expanduser('~/soderlinglab/run_utils'))
####### CHECK is the below load data enough- may be in hydra mode.l
# from train import load_data
from dataset import BinaryDataset, inference_custom_collate_fn
import yaml
from src.src_embeddings import parse_seqs

def main():
    config = f'/hpc/group/soderlinglab/user/pooja/projects/Biotin/src/configs/inference/inference_1.yaml' # /hpc/group/soderlinglab/user/pooja/projects/V5_proteomics/data/11087_SupplementalData_093025.xlsx
    with open(config, 'r') as file:
        data = yaml.safe_load(file)
    print('my config: \n', data)
    infer(data)
    return

def load_data(config_input):
    file =config_input['inference_file']
    ID_col = config_input['ID_col']
    seq_col = config_input['seq_col']
    positives_col = config_input['positives_col']
    embeddings_path = config_input['embeddings_path']
    esm_engine = ESMembedding(seqfile=file, IDcol=ID_col, seqcol=seq_col)
    dataset = BinaryDataset(file, ID_col, seq_col, positives_col, embeddings_path, esm_engine)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=inference_custom_collate_fn)
    return dataloader

@torch.no_grad()
def infer(config_input):
    ## load data
    dataloader = load_data(config_input)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ## load model
    model_path = config_input['model_path']
    print('model_path: ', Path(model_path).stem)
    model_state = torch.load(model_path, weights_only=True)
    model = Model6()
    model.load_state_dict(model_state)
    model.to(DEVICE)
    model.eval()
    pnc = Counter()
    
    PROJECT = config_input.get('project', str(model_path).replace('/', '_')) + '_' + str(Path(model_path).parent.stem)
    DATASET_NAME = str(Path(config_input['inference_file']).stem)
    RUN = wandb.init(
        entity="soderlinglab-dukecellbio",
        project=PROJECT,
        id=Path(config_input['inference_file']).stem,         ## unique identifier, make dataset
        job_type="inference",
        name= Path(config_input['inference_file']).stem, ## visual name in wandb
        group= Path(model_path).parent.stem, ## set of params of model for modelX
        config=config_input
    )
    labels = []
    preds = []

    for i, batch in enumerate(dataloader):
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
    acc, auroc, aupr, CM, recall, precision, barplot, heatmap = compute(labels, preds, threshold=0.5, method='inference')
    test_metrics = {'model_path': str(model_path),
                    'dataset': DATASET_NAME,
                    'test_accuracy': float(acc),
                   'test_auroc': float(auroc),
                   'test_aupr': float(aupr), ## cm
                   'pnc': safe_pnc}
    save_test_path = Path(f'/cwork/pkp14/Biotin/models/{PROJECT}/{DATASET_NAME}')
    save_test_path.mkdir(exist_ok=True, parents=True)
    with open(save_test_path / 'test_metrics.json', mode='w') as f:
        json.dump(test_metrics, f ,indent=4)
    df.to_csv(save_test_path / 'test_preds.csv', index=False)
    
    print(f'test metrics written to {save_test_path}')
    art = wandb.Artifact(name=f"{DATASET_NAME}_evaluation", type="eval", metadata=test_metrics)
    art.add_file(str(save_test_path / 'test_metrics.json'))
    art.add_file(str(save_test_path / 'test_preds.csv'))
    
    ## log wandb artifacts
    artifact_bundle =RUN.log_artifact(art)
    wandb.log({'confusion_matrix': wandb.Image(heatmap)})
    wandb.log({'metrics_bar_plot': wandb.Image(barplot)})
    artifact_bundle.wait()                 # ensure artifact upload completes
    print("Run URL:", RUN.url)    # sa
    wandb.finish()                # flush everything
    return

if __name__ == '__main__':
    main()
