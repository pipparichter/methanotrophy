'''A script which creates a CountMatrix object, pickles the result, and saves the serialized object.'''

import sys
sys.path.append('/home/prichter/Documents/trophy/src')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from matrix import CountMatrix, _reformat_csv
from transform import * 
import pickle
import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='The path to the CSV file containing the count data.', type=str)
    parser.add_argument('output', help='The path specifying where to write the pickled CountMatrix.', type=str)
    parser.add_argument('-m', '--metadata', help='The path to the CSV file containing sample metadata.', type=str, default=None)
    parser.add_argument('-f', '--filter-read-depth', help='The minimum allowed read depth per sample.', type=int, default=None)
    parser.add_argument('-n', '--normalize', help='The normalization method to use.', type=str, default=None, choices=['css', 'rar', 'clr'])
    parser.add_argument('-l', '--level', help='The taxonomic level for merging the ASVs. If none is given, ASV-level resolution is maintained.', type=str, default='asv', choices=['asv', 'species', 'genus', 'family', 'order', 'class', 'domain', 'phylum'])
    
    args = parser.parse_args()

    # Perform some checks on the input arguments and files. 
    input_file_type = args.input.split('.')[-1]
    output_file_type = args.output.split('.')[-1]

    assert input_file_type == 'csv', f'Unsupported input file type {input_file_type}. Must be a CSV file.'
    assert output_file_type == 'mtx', f'Unsupported output file type {input_file_type}. Must be a Matrix file.'
    
    cols = pd.read_csv(args.input, nrows=2).columns

    M = CountMatrix(level=args.level).from_pandas(_reformat_csv(args.input, level=args.level))
    M = M.filter_empty_cols() # Remove any empty columns from the DataFrame.

    # If metadata is specified, load into the CountMatrix object. 
    if args.metadata is not None:
        M.load_metadata(args.metadata)
    # Only keep samples which meet the minimum read depth requirement. 
    if args.filter_read_depth is not None:
        M.filter_read_depth(args.filter_read_depth)

    # If a normalization approach is specified, normalize the resulting matrix. 
    if args.normalize is not None:
        # Initialize the appropriate normalizer. 
        if args.normalize == 'rar':
            normalizer = Rarefaction()
        elif args.normalize == 'css':
            normalizer = ConstantSumScaling()
        elif args.normalize == 'clr':
            normalizer = CenteredLogRatioTransformation()
        normalizer(M) 

    # Save the CountMatrix to a pickle file. 
    with open(args.output, 'wb') as f:
        pickle.dump(M, f)



