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

## biotin locs are based on 1 indexing. transform for python 0 indexing
class LysineDataset(Dataset):
    def __init__(self, file, ID_col, biotin_locs_col, seq_col, embeddings_path):
        self.df = pd.read_csv(file)
        self.df = self.df.drop(columns=list(filter(lambda x: x.startswith('Unnamed'), self.df.columns)))
        self.ID_col = ID_col
        self.biotin_locs_col = biotin_locs_col
        self.seq_col = seq_col
        self.embeddings_path = embeddings_path
    
    def _check_residue(self, positions):
        ## correct indices
        valid_pos = set()
        for i in positions:
            if self.seq[i].upper() == 'K':
                valid_pos.add(int(i))
            else:
                continue
        return valid_pos
        
    def __len__(self):
        return len(self.df[self.ID_col].str.split(';').str[0].values)
        
    def __getitem__(self, idx):
        self.ID = re.split(r'[;,]', self.df.iloc[idx][self.ID_col])[0]
        self.seq = self.df.iloc[idx][self.seq_col]
        self.embedding = self._pull_embedding(self.ID)
        pos = ast.literal_eval(self.df.iloc[idx][self.biotin_locs_col])
        self.positives = self._adjust_indexing(pos)
        self._get_negatives()
        
        ## check res
        self.positives = self._check_residue(self.positives)
        self.negatives = self._check_residue(self.negatives)
        
        # get vectors
        pos_vectors = self._get_vectors(self.positives)   
        neg_vectors = self._get_vectors(self.negatives)   
        
        # Combine all vectors and create labels
        all_vectors = torch.cat([pos_vectors, neg_vectors], dim=0)
        labels = torch.cat([torch.ones(len(pos_vectors)), 
                           torch.zeros(len(neg_vectors))])  
        return all_vectors, labels, len(pos_vectors)
    
    def _adjust_indexing(self, l):
        return set(list(map(lambda x: int(x - 1), l)))
    
    def _get_negatives(self):
        self.lysine_locs = set([x.start() for x in re.finditer('K', self.seq)])
        self.negatives = self.lysine_locs - self.positives
        if not self.negatives or len(self.negatives) < len(self.positives):
            self.sample_random_negatives
        return
    
    def sample_random_negatives(self):
        n_negs = len(self.positives)
        negs = set(range(len(self.seq))) - self.positives
        extra_negs= np.random.choice(negs, size=n_negs, replace=False)
        return self.negatives.update(extra_negs)
    
    def _get_vectors(self, positions):
        if not positions:
            print(self.ID, positions, self.lysine_locs, self.positives)
        myembeds = []
        for i in positions:
            try:
                v = self.embedding.select(0, i)
                myembeds.append(v)
            except Exception as e:
                print(f'Unable to get the embedding {i}, error {e}, for protein {self.ID}')
        return torch.stack(myembeds)
    
    def _pull_embedding(self, seq_id):
        with h5py.File(Path(self.embeddings_path) / f'{seq_id}.h5', 'r') as f:  # Added 'r' mode
            embdg = torch.tensor(f[f'{seq_id}_representation'][:])
        return embdg
