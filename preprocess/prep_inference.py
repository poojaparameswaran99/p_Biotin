import pandas as pd
import numpy as np 
import os 
import sys
from pathlib import Path
from utils import prepare_peptide
sys.path.append(os.path.expanduser('~/soderlinglab/utils'))
from data_mapping.IDoperations import retrieve_annotation
from seqs.idseqio import get_seqs


def main():
    in_file= '../data/11101_SupplementalData_Normalized_101025.xlsx'
    ex = pd.ExcelFile(in_file)
    
    df3 = ex.parse(3)
    df3 = df3.sort_values(by='Accession').reset_index(drop=True)
    
    df4 = ex.parse(3)
    df4 = df4.sort_values(by='Accession').reset_index(drop=True)

    df= pd.concat([df3, df4], axis=0)
    df = df.drop_duplicates()
    ndf = df.pipe(prepare_peptide, 'Accession', 'Sequence', 'PeptidePosition')
    ndf['BiotinPosition'] = ndf['BiotinPosition'].apply(list)
    
    ## remove proteins already in train.
    ndf = ndf.pipe(remove_train_proteins, protein_col='Accession')
    ndf.to_csv(f'../data/new_inference/{Path(in_file).stem}.csv')
    
    return

def remove_train_proteins(df, protein_col):
    train_filepath = f'~/soderlinglab/user/pooja/projects/Biotin/data/train-val-test/train.csv'
    train_df = pd.read_csv(train_filepath)
    train_prots = train_df['Accession'].unique()
    df = df.query(f'{protein_col} not in @train_prots')
    return df

if __name__ == '__main__':
    main()
