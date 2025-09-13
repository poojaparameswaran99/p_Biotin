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
from itertools.chain import from_iterable
sys.path.append(os.path.expanduser('~/soderlinglab/run_utils/src'))
# from src_embeddings import parse_seqs

## things to do 
# update dataset configs
# fix train loop
# run on subset of data

class BinaryDataset(Dataset):
    def __init__(self, file, ID_col, seq_col, positives_col, embeddings_path):
        self.df = pd.read_csv(file)
        self.df[positives_col] = self.df[positives_col].apply(eval)
        self.ID_col = ID_col
        self.seq_col = seq_col
        self.positives_col = positives_col
        
    def _pull_embedding(self, seq_id):
        file_path = Path(self.embeddings_path) / f"{seq_id}.h5"
#         if not file_path.exists():
#             self._get_esm_embedding()
        with h5py.File(Path(self.embeddings_path) / f'{seq_id}.h5', 'r') as f:  # Added 'r' mode
            embdg = torch.tensor(f[f'{seq_id}_representation'][:])
        assert len(self.seq) == len(embdg)
        return embdg
    
    def _search_seq(self, seq, residue='K'):
        all_lysines = list(re.finditer(residue, seq))
        positions = [x.start() for x in all_lysines]
        return positions
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        seq = self.df.loc[idx, self.seq_col]
        positions = self._search_seq(seq)
        positives = self.df.loc[idx, self.positives_col]
        negatives = set(positiions) - set(positives)
        paired_res_labels = [(r, 1) for r in positions if r in positives else (r, 0)]
        pr = sorted(paired_res_labels, key=lambda x: x[0])
        residues, labels = zip(*pr)
        return residues, labels

def inference_custom_collate_fn(batch):
    vectors, labels = zip(*batch)
    vectors = from_iterable(vectors)
    labels = from_iterable(labels)
    return vectors, labels


class ContrastiveDataset(Dataset):
    def __init__(self, file, p_ID_col, a_ID_col, n_ID_col, p_idx_col, a_idx_col, n_idx_col, seq_col, embeddings_path):
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
        assert seq[ridx].upper() == 'K'; f'{ID} problematic, residue {ridx} not K! {seq[ridx]}'
        return
    
    def _get_vector(self,ID, seq, seq_embedding, ridx):
        self._check_residue(ID, seq, ridx)
        v = seq_embedding.select(0, ridx)
        return v
    
    def __getitem__(self, idx):
        seq =df.loc[idx, self.seq_col]
        ID = df.loc[idx, self.a_ID_col]
        idxs = [
            df.loc[idx, self.p_idx],
            df.loc[idx, self.a_idx],
            df.loc[idx, self.n_idx],
        ]
        seq_embedding = self._pull_embedding(ID)
        vectors = torch.stack([self._get_vector(seq_embedding, cidx) for cidx in idxs])
        # unpack into scalars
        return vectors
    
    def _pull_embedding(self, seq_id):
        file_path = Path(self.embeddings_path) / f"{seq_id}.h5"
#         if not file_path.exists():
#             self._get_esm_embedding()
        with h5py.File(Path(self.embeddings_path) / f'{seq_id}.h5', 'r') as f:  # Added 'r' mode
            embdg = torch.tensor(f[f'{seq_id}_representation'][:])
        assert len(self.seq) == len(embdg)
        return embdg
    

class LysineNegsDataset(Dataset):
    def __init__(self, file, ID_col, biotin_locs_col, seq_col, embeddings_path):
        self.df = pd.read_csv(file)
        self.df = self.df.drop(columns=list(filter(lambda x: x.startswith('Unnamed'), self.df.columns)))
        self.ID_col = ID_col
        self.biotin_locs_col = biotin_locs_col
        self.seq_col = seq_col
        self.embeddings_path = embeddings_path
#         self._get_esm_embedding(file, ID_col, seq_col, embeddings_path, model='esm2_15b')
        
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
        raw_pos = set(map(int, ast.literal_eval(self.df.iloc[idx][self.biotin_locs_col])))
        positives = self._adjust_indexing(raw_pos)
        negatives = self._get_negatives(positives)
        
        ## check res
        positives = self._check_residue(positives)
        negatives = self._check_residue(negatives)

        # get vectors
        pos_vectors = self._get_vectors(positives)    # [n, 5120]
        neg_vectors = self._get_vectors(negatives)
        
        # Combine all vectors and create labels
        all_vectors = torch.cat([pos_vectors, neg_vectors], dim=0)
        plab = torch.ones(pos_vectors.shape[0])
        nlab = torch.zeros(neg_vectors.shape[0])
        labels = torch.cat([plab, nlab]) 
#         assert len(plab) == len(nlab), f'pos:{len(positives)}; neg:{len(negatives)}; pos_vect:{len(pos_vectors)}; neg_vect:{len(neg_vectors)}, plab:{len(plab)}, nlab:{len(nlab)}'
        return all_vectors, labels, len(pos_vectors)
    
    def _adjust_indexing(self, l):
        return set(list(map(lambda x: int(x - 1), l)))
    
    def _get_negatives(self, p):
        self.lysine_locs = set([x.start() for x in re.finditer('K', self.seq)])
        n = sorted(list(self.lysine_locs - p))
        n = np.random.choice(n, size=len(p), replace=True)
        return n
    
    def _get_vectors(self, positions):
        if len(positions) == 0:
            raise ValueError("No positions found for this protein.")
        myembeds = []
        for i in positions:
            try:
                v = self.embedding.select(0, i)
                myembeds.append(v)
            except Exception as e:
                print(f'Unable to get the embedding {i}, error {e}, for protein {self.ID}')
        return torch.stack(myembeds)
    
    def _pull_embedding(self, seq_id):
        file_path = Path(self.embeddings_path) / f"{seq_id}.h5"
#         if not file_path.exists():
#             self._get_esm_embedding()
        with h5py.File(Path(self.embeddings_path) / f'{seq_id}.h5', 'r') as f:  # Added 'r' mode
            embdg = torch.tensor(f[f'{seq_id}_representation'][:])
        assert len(self.seq) == len(embdg)
        return embdg
    
    def _get_esm_embedding(self,file, ID_col, seq_col, embeddings_path, model='esm2_15b'):
        cfg = {
        "esm": model,
        "seqfile": file,
        "IDcol": ID_col,
        "seqcol": seq_col,
        "average_pool": False,
        "seqtype": "protein",
        "project_name": "Biotin",
        "overwrite": False,
        "output_dir": "/cwork/pkp14/${project_name}/${esm}_embeds",
        }
        parse_seqs(cfg)
        return



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
        raw_pos = set(map(int, ast.literal_eval(self.df.iloc[idx][self.biotin_locs_col])))
        positives = self._adjust_indexing(raw_pos)
        negatives = self._get_negatives(positives)
        
        ## check res
        positives = self._check_residue(positives)
#         self.negatives = self._check_residue(self.negatives)

        # get vectors
        pos_vectors = self._get_vectors(positives)    # [n, 5120]
        neg_vectors = self._get_vectors(negatives)
        
        # Combine all vectors and create labels
        all_vectors = torch.cat([pos_vectors, neg_vectors], dim=0)
        plab = torch.ones(pos_vectors.shape[0])
        nlab = torch.zeros(neg_vectors.shape[0])
        labels = torch.cat([plab, nlab]) 
        assert len(plab) == len(nlab), f'pos:{len(positives)}; neg:{len(negatives)}; pos_vect:{len(pos_vectors)}; neg_vect:{len(neg_vectors)}, plab:{len(plab)}, nlab:{len(nlab)}'
        return all_vectors, labels, len(pos_vectors)
    
    def _adjust_indexing(self, l):
        return set(list(map(lambda x: int(x - 1), l)))
    
    def _get_negatives(self, p):
        self.lysine_locs = set([x.start() for x in re.finditer('K', self.seq)])
        n = sorted(list(self.lysine_locs - p))
        n = np.random.choice(n, size=len(p), replace=True)
        if len(set(n)) < len(p):
            n = self.sample_random_negatives(set(n), p)
        return n
    
    def sample_random_negatives(self, n, p):
        negs = set(range(len(self.seq))) - p -n
        extra_negs= np.random.choice(list(negs), size=len(p)-len(n), replace=False)
        n.update(extra_negs)
        return n
    
    def _get_vectors(self, positions):
        if len(positions) == 0:
            raise ValueError("No positions found for this protein.")
        myembeds = []
        for i in positions:
            try:
                v = self.embedding.select(0, i)
                myembeds.append(v)
            except Exception as e:
                print(f'Unable to get the embedding {i}, error {e}, for protein {self.ID}')
        return torch.stack(myembeds)
    
    def _pull_embedding(self, seq_id):
        file_path = Path(self.embeddings_path) / f"{seq_id}.h5"
#         if not file_path.exists():
#             self._get_esm_embedding()
        with h5py.File(Path(self.embeddings_path) / f'{seq_id}.h5', 'r') as f:  # Added 'r' mode
            embdg = torch.tensor(f[f'{seq_id}_representation'][:])
        assert len(self.seq) == len(embdg)
        return embdg

#     def _get_esm_embedding(self,model='esm2_15b'):
        
#         return
 