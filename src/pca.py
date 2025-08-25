import sys
import os 
import pandas as pd
import numpy as np 
sys.path.append(os.path.expanduser('~/soderlinglab/utils'))
from sklearn.decomposition import PCA
import h5py
from pathlib import Path
import pickle

def main():
    embeddings_path = f'/cwork/pkp14/Biotin/esm2_15b_embeds'
    train_file = f'/hpc/group/soderlinglab/user/pooja/projects/Biotin/data/train-val-test/train.csv'
    validation_file =  f'/hpc/group/soderlinglab/user/pooja/projects/Biotin/data/train-val-test/validation.csv'
    test_file =  f'/hpc/group/soderlinglab/user/pooja/projects/Biotin/data/train-val-test/test.csv'
    pca_fetures_path = f'/cwork/pkp14/Biotin/esm2_15b_embeds'
    Path(pca_fetures_path).mkdir(exist_ok=True)
    ## read in train file to train pca
    train = pd.read_csv(train_file).iloc[:100]
    pca= fit_pca(train, 'Accession', embeddings_path)
    Xtrain = transform_input(pca, train_file, 'Accession', embeddings_path)
    # transform_input(pca, )
    return

def fit_pca(df, id_col, embeddings_path):
    data = load_embeddings(df[id_col].str.split(';').str[0].values, embeddings_path)
    pca = PCA(n_components= 100, svd_solver='randomized')
    pca.fit((list(data.values()))) # n_Samples, n_Features, else transpose
    print('pca componenets', pca.components_, 'size', pca.components_.shape)
    print('pca explained variance', pca.explained_variance_, 'size', pca.explained_variance_.shape)
    return pca

def transform_input(pca, filepath, id_col, embeddings_path):
    IDS = pd.read_csv(filepath)[id_col].iloc[:100].unique()
    data = load_embeddings(IDS, embeddings_path)
    print(data.values())
    Xt = pca.transform(np.array(list(data.values())))
    print(Xt, '\n\n\n', Xt[0])
    return Xt

def load_embeddings(IDS, embeddings_path):
    data = {}
    for ID in IDS:
        data[ID] = h5py.File( Path(embeddings_path) / f'{ID}.h5', 'r')[f'{ID}_representation'][:]
    return data

if __name__ == '__main__':
    main()
