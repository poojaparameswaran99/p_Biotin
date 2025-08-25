import pandas as pd 
import numpy as np 
import torch
import os 
import sys 
sys.path.append(os.path.expanduser('~/soderlinglab/utils'))
from collections import Counter
import json 


RANDOM_SEED = 123
VAL_PERCENT = 0.125
TEST_PERCENT = 0.1

def main():
    df = pd.read_csv('/hpc/group/soderlinglab/user/pooja/projects/Biotin/data/Biotin_protein_data_detailed.csv')
    ## cols to convert to lists
    df = df.drop(columns=[x for x in df.columns if x.startswith('Unnamed')])
    cols = ['BiotinPosition', 'Extracellular', 'Intracellular']
    df = df.pipe(convert_to_list_or_string, cols)
    df = df.pipe(check_residue, 'BiotinPosition', 'seq')
    df = df.query('Accession !="A2ASS6"')
    df = df.dropna()
    out=split_train_val_test_ind(df)
    splits = df.pipe(split, out)
    return

def split(df, dict_indices):
    splits = {}
    for sp, ind in dict_indices.items():
        splits[sp] = df.loc[list(ind)]
        df = df.drop(columns=[x for x in df.columns if any(x.startswith(y) for y in ['Unnamed', 'EC', 'IC', 'Order'])])
        print(sp,splits[sp].shape )
        splits[sp].to_csv(f'~/soderlinglab/user/pooja/projects/Biotin/data/train-val-test/{sp}.csv',index=False)
    return splits

def check_domain(domain):
    try:
        evaluated = eval(domain)
        return isinstance(evaluated, list) and len(evaluated) > 0
    except:
        return False

    df['seq'] = df['seq'].str.pad(width=maxlen, side='left', fillchar=pad_token)
    return df

def check_residue(df, biotin_col, seq_col):
    for i in range(df.shape[0]):
        post = df.iloc[i][biotin_col]
        seq = df.iloc[i][seq_col]
        valid_pos = set()
        ## need to subtract pythonic index
        for p in post:
            pa = int(p) -1
            if pa < len(seq) and seq[pa].upper() == 'K':
                valid_pos.add(int(p))
            else:
                continue
#         print(valid_pos)
        df.at[i, biotin_col] = sorted(valid_pos) if valid_pos else np.nan
    return df

def remove_by_feature(df, feature_col, percent_to_r = 0.25):
    fdf = df.query(f'{feature_col} == True')
    i_to_r = np.random.choice(fdf.index, int(np.ceil(percent_to_r*len(fdf))))
    return list(i_to_r)

def convert_to_list_or_string(df, columns):
    """
    Convert specified columns to lists if possible using eval(), 
    otherwise keep as strings.
    """
    def safe_eval_to_list(x):
        try:
            if isinstance(x, list):
                return x
            if hasattr(x, '__iter__') and not isinstance(x, str):
                return list(x)
            evaluated = eval(str(x))
            return evaluated if isinstance(evaluated, list) else str(x)
        except:
            return str(x)
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(safe_eval_to_list)
    return df

def split_train_val_test_ind(df):
    np.random.seed(123)
    ## better to keep accessions or 
    ### feature removal, total removal
    test_val_remove = {'test': (0.10), 'val': (0.125)}
    data_indices = {'train': set(), 'test': set(), 'val': set()}
    ## assign all single biotin proteins to test
    single_biotin_locs_index = df[df['BiotinPosition'].apply(len) <=1].index.tolist()
    data_indices['test'].update(single_biotin_locs_index)
    for split, p in test_val_remove.items():
        n_to_r = int(np.ceil(p*len(df)))
        ## remove extracellular data
        data_indices[split].update(df.pipe(remove_by_feature, 'EC'))
        print(f'len of split {split} after ec', len(data_indices[split]))
        data_indices[split].update(df.pipe(remove_by_feature, 'IC'))
        print(f'len of split {split} after ic', len(data_indices[split]))
        data_indices[split].update(np.random.choice(df.index, n_to_r))
        print(f'len of split {split} after random removal', len(data_indices[split]))
        ## remove the appropriate indices
        df = df.drop(index = data_indices[split])
    data_indices['train'].update(df.index)
    return data_indices

if __name__ == "__main__":
    main()
