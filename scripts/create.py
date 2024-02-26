'''A script which creates a CountMatrix object, pickles the result, and saves the serialized object.'''

import sys
sys.path.append('../src')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from matrix import *
import pickle
import argparse


def taxonomy_in_cols(cols:List[str]) -> bool:
    '''Confirm that a taxonomic level is provided in the CSV file.'''
    taxonomy_cols_in_csv_file = [col for col in cols_in_csv_file if col in TaxonomyMatrix.levels]
    return len(taxonomy_cols_in_csv_file) > 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='The path to the CSV file containing the count data.', type=str, required=True)
    parser.add_argument('output', help='The path specifying where to write the pickled CountMatrix.', type=str, required=True)
    parser.add_argument('-m', '--metadata', help='The path to the CSV file containing sample metadata.', type=str, default=None)
    parser.add_argument('-f', '--filter-read-depth', help='The minimum allowed read depth per sample.', type=int, default=None)
    parser.add_argument('-n', '--normalize', help='The normalization method to use.', type=str, default=None, choices=['css', 'rar', 'clr'])
    parser.add_argument('-l', '--level', help='The taxonomic level for merging the ASVs. If none is given, ASV-level resolution is maintained.', type=str, default='asv', choices=['species', 'genus', 'family', 'order', 'class', 'domain', 'phylum'])
    
    args = parser.parse_args()

    # Perform some checks on the input arguments and files. 
    input_file_type = args.input.split('.')[-1]
    output_file_type = args.input.split('.')[-1]

    assert input_file_type == 'csv', f'Unsupported input file type {input_file_type}. Must be a CSV file.'
    assert output_file_type == 'mtx', f'Unsupported output file type {input_file_type}. Must be a Matrix file.'
    
    cols = pd.read_csv(args.input, nrows=2).columns

    M = CountMatrix(level=args.level)
    M.read_csv(args.input)  
    # If metadata is specified, load into the CountMatrix object. 
    if args.metadata is not None:
        M.load_metadata(args.metadata)
    # Only keep samples which meet the minimum read depth requirement. 
    if args.filter_read_depth is not None:
        M.filter_read_depth(args.filter_read_depth)
    # If a normalization approach is specified, normalize the resulting matrix. 
    if args.normalize is not None:
        M.normalize(args.normalize)

    # Save the CountMatrix to a pickle file. 
    with open(args.out_path, 'wb') as f:
        pickle.dump(M, f)



