'''A script for creating an AsvMatrix object from a CSV file containing sample labels, count information, and ASV 
categorizations, as well as any other metadata.'''

import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
from matrix import *
import pickle
import argparse

def asv_matrix_from_df(df:pd.DataFrame) -> pd.DataFrame:
    '''Convert the DataFrame containing the sample metadata and ASV counts into an AsvMatrix object.''' 
    # Accumulate all of the metadata into a separate DataFrame. 
    metadata = df[[col for col in df.columns if col not in ['count']]]

    df = df[['serial_code', 'asv', 'count']]
    df = df.groupby(by=['serial_code', 'asv']).sum()
    df = df.reset_index() # Converts the multi-level index to categorical columns. 
    df = df.pivot(columns=['asv'], index=['serial_code'], values=['count'])
    # Reset column labels, which were weird because of the multi-indexing. 
    df.columns = df.columns.get_level_values('asv').values

    return AsvMatrix(df, metadata=metadata)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', help='The path to the CSV file containing the ASV count data.', type=str)
    parser.add_argument('out_path', help='The path specifying where to write the pickled AsvMatrix.', type=str)
    parser.add_argument('-f', '--filter-read-depth', help='The minimum allowed read depth per sample.', type=int, default=5000)
    
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)
    asv_matrix = asv_matrix_from_df(df)
    
    # Only keep samples which meet the minimum read depth requirement. 
    asv_matrix.filter_read_depth(args.filter_read_depth)
    
    # Save the ASV matrix in a pickle file. 
    with open(args.out_path, 'wb') as f:
        pickle.dump(asv_matrix, f)



