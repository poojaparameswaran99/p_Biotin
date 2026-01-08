import pandas as pd
import numpy as np 
import os 
import sys
import itertools
import functools
from torch.utils.data import Dataset, DataLoader
import re
import ast
from pathlib import Path
import h5py
import torch
import subprocess
from itertools import chain
sys.path.append(os.path.expanduser('~/soderlinglab/run_utils/src'))
from src_embeddings import parse_seqs

## things to do 
# update dataset configs
# fix train loop
# run on subset of data

class BinaryDataset(Dataset):
    def __init__(self, file, ID_col, seq_col, positives_col, embeddings_path, esm_engine):
        self.df = pd.read_csv(file)
        self.df[positives_col] = self.df[positives_col].apply(eval)
        self.ID_col = ID_col
        self.seq_col = seq_col
        self.positives_col = positives_col
        self.embeddings_path= embeddings_path
        self.esm_engine = esm_engine
        
        
    def _pull_embedding(self, seq_id, seq):
        file_path = Path(self.embeddings_path) / f"{seq_id}.h5"
        if not file_path.exists():
            self.embedding_engine.make_embedding()
        with h5py.File(Path(self.embeddings_path) / f'{seq_id}.h5', 'r') as f:  # Added 'r' mode
            embdg = torch.tensor(f[f'{seq_id}_representation'][:])
        assert len(seq) == len(embdg)
        return embdg

    def _search_seq(self, seq, residue='K'):
        all_lysines = list(re.finditer(residue, seq))
        positions = [x.start() for x in all_lysines]
        return positions
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        seq = self.df.loc[idx, self.seq_col]
        ID = self.df.loc[idx, self.ID_col]
        seq_embdg = self._pull_embedding(ID, seq)
        positions = self._search_seq(seq)
        positives = self.df.loc[idx, self.positives_col]
        negatives = set(positions) - set(positives)
        paired_res_labels = [(r, 1) if r in positives else (r, 0) for r in positions]
        pr0 = sorted(paired_res_labels, key=lambda x: x[0])
        pr1 = list(map(lambda x: (seq_embdg[x[0]], x[1]), pr0))
        residues, labels = zip(*pr1)
        return residues, torch.tensor(labels)

def inference_custom_collate_fn(batch):
    vectors, labels = zip(*batch)
    vectors0 = list(chain.from_iterable(vectors))
    labels0 = list(chain.from_iterable(labels))
    vectors1 =  torch.stack(vectors0, dim=0)
    labels1 = torch.stack(labels0, dim=0)
    return vectors1, labels1


class ContrastiveDataset(Dataset):
    def __init__(self, file, p_ID_col, a_ID_col, n_ID_col, p_idx_col, a_idx_col, n_idx_col, seq_col, embeddings_path, esm_engine):
        self.df = pd.read_csv(file)
        self.df = self.df.drop(columns=list(filter(lambda x: x.startswith('Unnamed'), self.df.columns)))
        self.p_ID_col = p_ID_col
        self.a_ID_col = a_ID_col
        self.n_ID_col = n_ID_col
        self.p_idx_col = p_idx_col
        self.a_idx_col = a_idx_col
        self.n_idx_col = n_idx_col
        self.seq_col = seq_col
        self.embeddings_path = embeddings_path
    
    def __len__(self):
        return self.df.shape[0]
    
    def _check_residue(self, ID, seq, ridx):
        assert seq[ridx].upper() == 'K', f'{ID} problematic, residue {ridx} not K! {seq[ridx]}'
        return
    
    def _get_vector(self,ID, seq, seq_embedding, ridx):
        self._check_residue(ID, seq, ridx)
        v = seq_embedding.select(0, ridx)
        return v
    
    def __getitem__(self, idx):
        seq = self.df.loc[idx, self.seq_col]
        ID = self.df.loc[idx, self.a_ID_col]
        idxs = [
            self.df.loc[idx, self.p_idx_col],
            self.df.loc[idx, self.a_idx_col],
            self.df.loc[idx, self.n_idx_col],
        ]
        seq_embedding = self._pull_embedding(ID, seq)
        vectors = torch.stack([self._get_vector(ID, seq, seq_embedding, cidx) for cidx in idxs])
        # unpack into scalars
        return vectors
    
    def _pull_embedding(self, seq_id, seq):
        file_path = Path(self.embeddings_path) / f"{seq_id}.h5"
#         if not file_path.exists():
#             self._get_esm_embedding()
        with h5py.File(Path(self.embeddings_path) / f'{seq_id}.h5', 'r') as f:  # Added 'r' mode
            embdg = torch.tensor(f[f'{seq_id}_representation'][:])
        assert len(seq) == len(embdg)
        return embdg
    

class ESMembedding:
    def __init__(self, file, IDcol, seqcol,
                 esm='esm2_15b', average_pool=False,
                 seqtype='Protein', project_name='Biotin',
                 overwrite=False):
        cfg = {'esm': esm,
              'seqfile': file,
               'IDcol': IDcol,
               'seqcol': seqcol,
               'average_pool': average_pool,
               'seqtype': seqtype,
               'project_name': project_name,
               'overwrite': overwrite,
               'output_dir': f'/cwork/pkp14/{project_name}/{esm}_embeds'
              }
        
        self.cfg = cfg
        
    def make_embedding(self):
        parse_seqs(self.cfg)
        return
    
