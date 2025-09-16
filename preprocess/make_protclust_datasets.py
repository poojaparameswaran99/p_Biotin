import pandas as pd 
import numpy as np 
import os 
import sys
import ast
import re
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser('~/soderlinglab/tools/protclust'))
from protclust import clean, cluster, split, set_verbosity, milp_split
from make_train_pairs import parse

SEED=123
def main():
    np.random.seed(123)
    file = '~/soderlinglab/user/pooja/projects/Biotin/data/Protein_w_BiotinSites.csv'
    localcluster_and_write_datasets(file)
    return

def localcluster_and_write_datasets(file):
    set_verbosity(verbose=True)
    
    ## read in data
    df = pd.read_csv(file)
    
    ## clean data w protclust clean
    dfc = clean(df, sequence_col='seq')
    
    ## cluster df
    clustered_df = cluster(dfc, sequence_col='seq', id_col='Accession')
    clustered_df.to_csv('tmp_data/cluster_check.csv', index=False)
    
    ## train , test split, independent var=nBiotins
    train_df, test_df = milp_split(clustered_df, group_col='cluster_representative', test_size=0.28, 
                                  balance_cols='nBiotinLocs')
    ## train, val split, indepednent var = nBiotins
    train_df, val_df = milp_split(train_df, group_col='cluster_representative', test_size=0.15, 
                                  balance_cols='nBiotinLocs')
    
    ## chck train, val, test splits
    tmp_df = pd.concat([train_df, val_df, test_df], axis=0)
    tmp_df.to_csv('tmp_data/cluster_check.csv', index=False)
    
    # move the singular biotin points out of train
    train_df, val_df,test_df = divvy_oneBiotinylation(train_df, val_df, test_df)
    
    ## write out
    for df, split in zip([train_df, val_df, test_df], ('train', 'val', 'test')):
        write_out(df, split)
    train_file = f'~/soderlinglab/user/pooja/projects/Biotin/data/train-val-test/train.csv'
    
    ## make contrastive training pairs

    parse(train_file, 'Accession','seq', 'BiotinPosition')
    print('training pairs made')
    return

def divvy_oneBiotinylation(train_df, val_df, test_df,test_out_percent=0.6, val_out_percent=0.4):
    inference_df = pd.concat([val_df, test_df], axis=0)
    ### samples with only one biotin in train
    train1Biotin_Index= sorted(train_df.query('nBiotinSites == 1').index)
    print(f'length of 1 biotin points in train')
    ## move 1 biotin samples from train to test
    inference_df = pd.concat([inference_df, train_df.loc[train1Biotin_Index]], axis=0)
    ## drop 1 biotin samples from train
    train_df = train_df.drop(index=train1Biotin_Index)
    ## put some samples from inference back into train
    # more than 1 bio site samples in inference
    infGT1Biotin_Index = sorted(inference_df.query('nBiotinSites >1').index)
    # sample length of samples pulled out == 1 site to put back into train
    GT1_put_back_in_train_Index = np.random.choice(infGT1Biotin_Index, size=len(train1Biotin_Index))
    ## add extra train samples
    train_df = pd.concat([train_df, inference_df.loc[GT1_put_back_in_train_Index]], axis=0)
    inference_df = inference_df.drop(index=GT1_put_back_in_train_Index)
    
    ## split inference set
    ntestout = len(inference_df) * test_out_percent
    nvalout = len(inference_df) * val_out_percent
    test_df = inference_df.sample(round(ntestout), random_state=123)
    val_df = inference_df.drop(test_df.index)
    data_checks(train_df, val_df, test_df)
    print('data checks done')
    return train_df, val_df, test_df

def write_out(df, split):
    df.to_csv(f'~/soderlinglab/user/pooja/projects/Biotin/data/train-val-test/{split}.csv', index=False)
    print(f'{split} written')
    return df

def cluster_diversity(train_df, val_df, test_df):
    '''Evaluates diversity between dataset with cluster n checks, looks at amt of sequence diversity between and within datasets'''
    train_n_clusters = train_df['cluster_representative'].nunique()
    test_n_clusters = test_df['cluster_representative'].nunique()
    val_n_clusters = val_df['cluster_representative'].nunique()
    print('### train ####')
    print(f'clusters: {train_n_clusters}, nproteins: {train_df["Accession"].nunique()}')
    print(f'Cluster diversity: {train_df.groupby("cluster_representative")["Accession"].nunique().value_counts()}')
    print('### val #####')
    print(f'clusters: {val_n_clusters}, nproteins: {val_df["Accession"].nunique()}')
    print(f'Cluster diversity: {val_df.groupby("cluster_representative")["Accession"].nunique().value_counts()}')
    print('#### test #####')
    print(f'clusters: {test_n_clusters}, nproteins: {test_df["Accession"].nunique()}')
    print(f'Cluster diversity: {test_df.groupby("cluster_representative")["Accession"].nunique().value_counts()}')
    return

def plot_data(train_df, val_df, test_df):
    train_nbiotins = train_df['BiotinPosition'].apply(len)
    train_nbiotin_binned = pd.cut(train_nbiotins.values, bins=5)

    ## val
    val_nbiotins = val_df['BiotinPosition'].apply(len)

    ## test
    test_nbiotins = test_df['BiotinPosition'].apply(len)
    test_nbiotin_binned = pd.cut(test_nbiotins.values, bins=5)


    fig, ax = plt.subplots(2,2, figsize=(9,9))
    train_nbiotins.value_counts().plot.bar(rot=60, color="b",  ax=ax[0,0], title='train dataset nBiotin per protein', xlabel='nBiotinSites',
                                               ylabel='Frequency')

    test_nbiotins.value_counts().plot.bar(rot=60, color="b",  ax=ax[0,1], title='test dataset nBiotin per protein', xlabel='nBiotinSites',
                                               ylabel='Frequency')

    val_nbiotins.value_counts().sort_values(ascending=False).plot.bar(rot=60, color='g', ax=ax[1,1], title='validation dataset nBiotin per protein', xlabel='nBiotinSites', 
                                         ylabel='Frequency')

    # sync all axes
    for axi in ax.flat:
        axi.set_xlim(ax[0,0].get_xlim())  # same x range
        axi.set_ylim(ax[0,0].get_ylim())  # same y range

    fig.text(0.5, 0.04, "nBiotinSites", ha="center")
    fig.text(0.04, 0.5, "Frequency", va="center", rotation="vertical")

    plt.tight_layout()
    plt.savefig(f'plots/biotin_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    return

def data_checks(train_df, val_df, test_df):
    ## check no training samples only have one Biotinylated location
    assert bool(train_df['BiotinPosition'].apply(len).eq(1).any()) is False
    train_proteins = train_df['Accession'].unique()
    val_proteins = val_df['Accession'].unique()
    test_proteins = test_df['Accession'].unique()
    n_total_prots = len(train_proteins) + len(val_proteins) + len(test_proteins)
    assert set(train_proteins) & set(test_proteins) == set()
    assert set(test_proteins) & set(val_proteins) == set()
    assert set(val_proteins) & set(train_proteins) == set()
    test_one_biotin = (test_df['BiotinPosition'].apply(len).eq(1).sum())
    val_one_biotin =  (val_df['BiotinPosition'].apply(len) == 1).sum()
    
    train_n_biotin = train_df['BiotinPosition'].apply(len).sum()
    test_n_biotin = test_df['BiotinPosition'].apply(len).sum()
    val_n_biotin = val_df['BiotinPosition'].apply(len).sum()
    n_total_biotin_sites = (train_n_biotin) + (test_n_biotin) + (val_n_biotin)
    ## check balances
    plot_data(train_df, val_df, test_df)
    print(f'Total amt of Biotinylation: {n_total_biotin_sites}')
    print(f'### Train Set ####')
    print(f'train size: {train_df.shape[0]/ n_total_prots}, Total Biotin Counts Sample Size: {train_n_biotin/ n_total_biotin_sites}')
    print(f'#### Test Set ####')
    print(f'Test size: {test_df.shape[0] / n_total_prots}, Total Biotin Counts Sample Size: {test_n_biotin/ n_total_biotin_sites}, '
          f'1BiotinSite Sample Size: {test_one_biotin}')
    print(f'#### Validation Set #####')
    print(f'Validation size: {val_df.shape[0]/ n_total_prots}, Total Biotin Counts Sample Size: {val_n_biotin/ n_total_biotin_sites}, '
          f' 1BiotinSite Sample Size: {val_one_biotin}')
    cluster_diversity(train_df, val_df, test_df)
    return

if __name__ == '__main__':
    main()
 