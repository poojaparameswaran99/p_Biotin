import requests
import os 
import sys
import pandas as pd 
import numpy as np
import re
from io import StringIO
sys.path.append(os.path.expanduser('~/soderlinglab/utils'))
from seqs.getdomains import run_analysis

def main():
    df = pd.read_csv('~/soderlinglab/user/pooja/projects/Biotin/data/Protein_w_BiotinSites.csv')
    search = ('topological domain', 'Extracellular', 'Extracellular')
    df = df.pipe(run_analysis, search, protein_col='Accession')
    search = ('topological domain', 'Cytoplasmic', 'Intracellular')
    df = df.pipe(run_analysis, search, protein_col='Accession')
    df.to_csv('domainsMapped_protein_w_biotinsites.csv', index=False)
    return

def get_data(df, protein_col):
    df['xml_data'] = df[protein_col].apply(get_xml)
    return

def get_family(response):
    pattern = r'\b\b\w+\s+family\b'
    matches = re.findall(pattern, response.text, re.IGNORECASE)
    return 

def get_xml(pid):
    URL = f'https://rest.uniprot.org/uniprotkb/{pid}.xml'
    r = requests.get(URL)
    if r.status_code == 200:
        return r
    return False

if __name__ == '__main__':
    main()
