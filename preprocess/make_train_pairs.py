import pandas as pd 
import numpy as np 
import os 
import sys
import ast
import re

SEED = 123
def main():
    np.random.seed(123)
    file = f'~/soderlinglab/user/pooja/projects/Biotin/data/train-val-test/train.csv'
    parse(file, 'Accession','seq', 'BiotinPosition')
    return

def parse(file, ID_col, seq_col, site_col):
    df = pd.read_csv(file)
    print('in parse, file read')
    df = df.pipe(eval_col, site_col).pipe(adjust_indexing, site_col).pipe(get_all_lysine_sites, 
                                                                                          seq_col).pipe(get_negatives, site_col, 'AllLysineSites')
    print(f'adjusted indexing')
    df = df.pipe(make_pairs, site_col, 'NegativeSites')
    print('parse, pairs made')
    write_out(df, ID_col, 'Pairs', seq_col)
    print('writing out')
    return

def eval_col(df, col_to_eval):
    intermediate = df[col_to_eval].apply(ast.literal_eval).apply(lambda x: sorted(set(x)))
    df[col_to_eval] = intermediate
    return df

def adjust_indexing(df, col_to_adjust):
    df[col_to_adjust] = df[col_to_adjust].apply(lambda xs: set([x-1 for x in xs]))
    return df

def get_all_lysine_sites(df, seq_col):
    df['AllLysineSites'] = df[seq_col].apply(lambda x: set({y.start() for y in re.finditer('K', x)}))
    return df

def random_derangement(arr, max_iter=100):
    if len(arr) <2:
        return np.random.permutation(list(arr))
    for i in range(max_iter):
        perm = np.random.permutation(list(arr))
        if np.all(perm != arr):   # true derangement: no fixed points
            return perm
    return perm

def write_out(df, ID_col, pairs_col, seq_col):
    rows = [[r[ID_col]]*3 + list(t) for _, r in df.iterrows() for t in r[pairs_col]]     
    odf = pd.DataFrame(rows, columns=['pos_ID', 'anchor_ID', 'neg_ID', 'pos_idx', 'anchor_idx', 'neg_idx'])
    seqmapping = df.set_index(ID_col)[seq_col].to_dict()
    odf['seq'] = odf['pos_ID'].map(seqmapping)
    odf[['pos_idx', 'anchor_idx', 'neg_idx']] = odf[['pos_idx', 'anchor_idx', 'neg_idx']].astype(int)
    odf.to_csv('~/soderlinglab/user/pooja/projects/Biotin/data/train-val-test/training_pairs.csv', index=False)
    return 

def make_pairs(df, biotin_sites_col, neg_sites_col):
    anchors = df[biotin_sites_col].apply(lambda x: list(x))
    print('anchors', anchors)
    positives = df.pipe(pull_positives, biotin_sites_col)
    print('positivews')
    negs = df.pipe(pull_negatives, neg_sites_col, biotin_sites_col)
    df['Pairs'] = [list(zip(p,a,n)) for p,a,n in zip(positives, anchors, negs)]
    return df ## return df

def pull_positives(df, biotin_sites_col):
    pos = df[biotin_sites_col].apply(lambda x: random_derangement(x))
    return pos

def pull_negatives(df, neg_sites_col, bio_sites_col):
    negs = df.apply(lambda x: np.random.choice(list(x[neg_sites_col]), size=len(x[bio_sites_col]), \
                                               replace=True if len(x[neg_sites_col]) < len(x[bio_sites_col]) else False), axis=1)
    return negs

def get_negatives(df, biotin_sites_col, all_lysine_sites_col):
    df['NegativeSites'] = df[all_lysine_sites_col] - df[biotin_sites_col]
    return df

if __name__ == '__main__':
    main()

