'''A script for generating a TaxonomyMatrix from an AsvMatrix in a pickle file. The script saves the resulting object
as a pickle file in a user-specified directory.'''

import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
from matrix import *
import pickle
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', help='The path to the CSV file containing the ASV count data.', type=str)
    parser.add_argument('out_path', help='The path specifying where to write the pickled TaxonomyNatrix.', type=str)
    parser.add_argument('-l', '--level', help='The taxonomic level for merging the ASVs.', type=str, default='phylum', choices=['species', 'genus', 'family', 'order', 'class', 'domain', 'phylum'])
    args = parser.parse_args()

    # Load the ASV matrix in from a pickle file. 
    with open(args.in_path, 'rb') as f:
        asv_matrix = pickle.load(f)

    taxnomy_matrix = asv_matrix.get_taxonomy_matrix(level=args.level)
    with open(args.out_path, 'wb') as f:
        pickle.dump(taxonomy_matrix, args.out_path)