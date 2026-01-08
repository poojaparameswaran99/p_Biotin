import pandas as pd 
import numpy as np 
from functools import partial
import os 
import sys 
import random
from operator import itemgetter
import re
sys.path.append(os.path.expanduser('~/soderlinglab/utils'))
from data_mapping.IDoperations import retrieve_annotation
from seqs.idseqio import get_seqs

def transform_data(df):
    ndf = df.groupby('Accession').agg({'BiotinPosition': set, 
                            'seq': 'first'})
    return ndf 

def prepare_peptide(df, id_col, peptide_col, peptide_position_col, biotin_keyword='[Biotin]'):
    ## clean up peptide
    df = get_PTMlocation(df, peptide_col, peptide_position_col, biotin_keyword)
    
    df = clean_peptide(df, peptide_col)
    
    ## map ids to sequence
    ids_mapped_to_seqs = get_seqs(df[id_col].unique())
    df['seq'] = df[id_col].map(ids_mapped_to_seqs)
    
    df = check_sites(df, 'BiotinPosition', 'seq')
    
    df = remove_seqs_len_threshold(df, 'seq', threshold=7000)
    df = transform_data(df)
    df.to_csv(f'check_data.csv')
    
    return df

def get_PTMlocation(df, peptide_col: str, peptide_position_col: str, biotin_keyword='[Biotin]'):
    ## assumes one match 
    biotin_loc=df[peptide_col].str.strip('_').str.find(biotin_keyword)-1 ## subtract to index the Lysine
    biotin_loc = biotin_loc.replace(-1, np.nan)
    ##python indexing
    df['BiotinPosition'] = df.loc[:, peptide_position_col].str.split(';|,|:').str[0].astype(int) + biotin_loc -1
    return df

def clean_peptide(df, peptide_col):
    df[peptide_col] = df[peptide_col].str.replace(r'\[(?!Biotin\])[^\]]*\]|_', '',regex=True)
    return df

def remove_seqs_len_threshold(df, seq_col, threshold=7000):
    indices =df[seq_col].str.len() > threshold
    oldshape = df.shape[0]
    df = df.loc[~indices]
    newshape = df.shape[0]
    print(f'{oldshape -newshape} have a length longer than threshold')
    return df

def is_Lysine(seq, pos):
    if int(pos) >= len(seq):
        return False
    return seq[int(pos)].upper() == 'K'

def check_sites(df, biotin_col, seq_col):
    df = df.dropna(subset=[biotin_col, seq_col])
    ## instantiate is_Lysine with 1 param w partial and then add extra parameter
    df = df[df.apply(lambda x: is_Lysine(x[seq_col], x[biotin_col]), axis=1)]
    return df

