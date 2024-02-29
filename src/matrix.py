import numpy as np
import pandas as pd
import itertools
from typing import NoReturn, List, Tuple
import os
import dask.dataframe 
# from tqdm.dask import TqdmCallback
from tqdm import tqdm
import subprocess
import time

# TODO: Maybe think of a better way of organizing sample versus taxonomy metadata. 


class Matrix():

    def __init__(self, df:pd.DataFrame=None) -> NoReturn:

        self.metadata = None

        if df is not None:
            self.matrix = df.values
            self.row_labels = df.index.values
            self.col_labels = df.columns.values

    def __getitem__(self, i):
        return self.matrix[i]        

    def __len__(self):
        return len(self.matrix)

    def transpose(self):
        '''Transpose the underlying data.'''

        row_labels, col_labels = self.row_labels, self.col_labels
        self.col_labels = row_labels
        self.row_labels = col_labels
        self.matrix = self.matrix.T
    
    def shape(self) -> Tuple[int, int]:
        '''Returns the shape of the underlying matrix.'''
        return self.matrix.shape

    def load_metadata(self, path:str):
        '''Read in sample metadata, and store as an attribute.'''
        # Get the indices such that each sample is included only once. This avoids reading in the entire array. 
        self.metadata = pd.read_csv(path, index_col=None) 


class DistanceMatrix(Matrix):

    def __init__(self, df:pd.DataFrame=None, metric:str=None):
        '''Initialize a DistanceMatrix object.'''
        super().__init__(df=df) # Initialize the parent class.
        self.metric = metric


class CountMatrix(Matrix):
    '''A matrix which contains observation counts in each cell. '''
    
    levels = ['phylum', 'class', 'order', 'genus', 'species', 'family', 'domain', 'asv']

    def __init__(self, df:pd.DataFrame=None, level:str='asv'):
        '''Initialize a CountMatrix object.'''
        super().__init__(df=df) # Initialize the parent class.
        # Remove any empty columns
        self.normalized = False

        assert level in CountMatrix.levels, 'CountMatrix.__init__: Specified level {level} is invalid.'
        self.level = level

        if df is not None:
            self.filter_empty_cols()

    def _read_csv_with_dask(self, path:str):
        '''Read a pandas DataFrame from a CSV file using Dask. This approach was found to be faster than loading the 
        CSV file in chunks.'''
        ddf = dask.dataframe.read_csv(path, usecols=[self.level, 'count', 'serial_code'], dtype={self.level:str, 'count':int, 'serial_code':int}, blocksize=25e6)
        ddf = ddf.groupby(by=['serial_code', self.level]).sum()
        ddf = ddf.reset_index() # Converts the multi-level index to categorical columns. 
        ddf = ddf.compute() # Not totally sure why compute needs to be called here. Converts the Dask DataFrame to a pandas DataFrame. 
        ddf = ddf.pivot_table(columns=self.level, index='serial_code', values='count')
        # Reset column labels, which were weird because of the multi-indexing. 
        ddf.columns = ddf.columns.get_level_values(self.level).values
        return ddf

    def read_csv(self, path:str, verbose:bool=False) -> NoReturn:
        '''Convert the DataFrame containing the sample metadata and ASV counts into an AsvMatrix object.'''
        assert self.level is not None, 'CountMatrix.read_csv: A level must be specified prior to loading in a CSV file. '
        ti = time.perf_counter()
        df = self._read_csv_with_dask(path)
        tf = time.perf_counter()
        if verbose: print(f'CountMatrix.read_csv: CSV file loaded into CountMatrix in {np.round(tf - ti, 2)} seconds.')
        self.__init__(df=df, level=self.level) # Run the initialization function with the DataFrame as input. 

    def to_csv(self, path:str) -> NoReturn:
        '''Convert a CountMatrix and associated metadata to a CSV file.'''
        # Use dask to process the underlying data for speed. This approach is substantially faster than writing in chunks.  
        ddf = pd.DataFrame(self.matrix, columns=self.col_labels)
        ddf['serial_code'] = self.row_labels
        ddf = dask.dataframe.from_pandas(ddf, npartitions=500)
        ddf = ddf.melt(id_vars=['serial_code'], var_name=self.level) 
        ddf = ddf.rename(columns={'value':'count'})

        ddf.to_csv(path, index=False)

    def get_chi_squared_distance_matrix(self) -> DistanceMatrix:
        '''Compute the chi-squared distance matrix between rows. 
        Used formula from https://link.springer.com/referenceworkentry/10.1007/978-0-387-32833-1_53.'''

        P = self.matrix / np.sum(self.matrix) # Get the relative frequency table by dividing entries by the grand total. 
        nrows, ncols = P.shape # Store number of rows and columns. 

        # Define row and column vector totals. 
        pi, pj = np.sum(P, axis=1, keepdims=True), np.sum(P, axis=0, keepdims=True)
        self.pi, self.pj = pi, pj # Store the summed up vectors in the matrix. 

        D = np.zeros((nrows, nrows)) # Initialize an empty distance matrix. 
        for a in range(P.shape[0]): # Iterate over the rows.
            rows = P
            row_a = rows[a][np.newaxis, :] # Grab the primary row for this iteration. Add a new axis for broadcasting, so it is [[x1, x2, ...]].
            d = np.square(row_a - rows) # Should have the same dimensions as P. 
            assert d.shape == P.shape, f'matrices.Matrix.get_chi_squared_distance_matrix: The vector d should be shape {P.shape}. Instead, shape is {d.shape}.'
            d = d / pj # Divide by the column totals. 
            d = np.sum(d, axis=1) # Sum up over the columns. This collapses the dimension. 
            d = np.sqrt(d) # Take the square root of the sum. 
            assert len(d) == nrows, f'matrices.Matrix.get_chi_squared_distance_matrix: The vector d should be size {nrows}. Instead, shape is {d.shape}.'
            D[a] = d # Store the computed distances in the matrix. 

        df = pd.DataFrame(D, columns=self.row_labels, index=self.row_labels)
        return DistanceMatrix(df=df, metric='chi-squared') 

    def get_bray_curtis_distance_matrix(self) -> DistanceMatrix:
        '''Calculate the Bray-Curtis similarity score for each pair of samples in the
        AsvMatrix object.'''

        # Vectorizing sped this up from ~2 minutes to ~4 seconds!
        D = np.zeros((len(self.row_labels), len(self.row_labels))) # Initialize an empty distance matrix. 
        s = self.matrix.sum(axis=1) # Compute the total counts in each sample. 
        for i in range(len(self.row_labels)):
            x = self.matrix[i] # Grab the primary sample.
            # If axis = 1, apply_along_axis will loop through each row and implement the function to the row. 
            c = np.apply_along_axis(lambda y : np.where(np.less(y, x), y, x), 1, self.matrix).sum(axis=1)
            D[i] = 1 - (2 * c) / (s + s[i])

        df = pd.DataFrame(D, columns=self.row_labels, index=self.row_labels)
        return DistanceMatrix(df, metric='bray-curtis')

    def get_metadata(self, field:str) -> pd.Series:
        '''Extract information from a particular field in the metadata attribute.
        
        :param field: The name of the metadata field to extract.
        :return: A pandas Series containing the metadata values. 
        '''
        # Some checks to make sure the flux data is present. 
        assert self.metadata is not None, 'CountMatrix.get_metadata: There is no metadata stored in the CountMatrix object.'
        assert field in self.metadata.columns, f'CountMatrix.get_metadata: There is no field {field} in the CountMatrix metadata.'

        # The metadata table contains an entry for each ASV group. This reduces the metadata to one entry per sample. 
        sample_metadata = self.metadata[['serial_code', field]].drop_duplicates(ignore_index=True)
        # Extract the data from the metadata table.
        return sample_metadata[field]

    def get_metadata_fields(self) -> List[str]:
        '''Return a list of the fields contained in the underlying metadata.'''
        assert self.metadata is not None, 'CountMatrix.get_metadata_fields: There is no metadata stored in the CountMatrix object.'
        return list(self.metadata.columns)

    def filter_empty_cols(self):
        '''Remove empty columns from the matrix. Empty columns can occur when a subset of the total
        data is loaded, and not all ASVs or Taxonomical categories are represented.'''

        non_empty_idxs = np.sum(self.matrix, axis=0) > 0

        self.col_labels = self.col_labels[non_empty_idxs]
        self.matrix = self.matrix[:, non_empty_idxs]

    def filter_read_depth(self, min_depth:int):
        '''Filter the matrix so that only samples with more than the specified number of reads are kept.
        
        :param min_depth: The minimum depth requirement for keeping a sample in the AsvMatrix. 
        '''
        depths = np.sum(self.matrix, axis=1)
        idxs = depths >= min_depth
        print(f'matrices.CountMatrix.filter_read_depth: Discarding {len(self.matrix) - np.sum(idxs)} samples with read depth less than {min_depth}.')

        self.matrix = self.matrix[idxs]
        self.row_labels = self.row_labels[idxs]
        # If metadata is present, make sure to drop filtered samples. 
        if self.metadata is not None:
            self.metadata = self.metadata[self.metadata.serial_code.isin(self.row_labels)]

    def sample(self, i:int, n:int, species_count_only:bool=False):

        s = self.matrix[i] # Get the sample from the matrix. 
        
        # It doesn't make sense to sample from an array of relative abundances, I think. 
        assert self.normalized == False, 'matrix.AsvMatrix.sample: The AsvMatrix has already been normalized.'
        assert n <= np.sum(s), f'matrix.AsvMatrix.sample: The sample size must be no greater than the total number of observations ({np.sum(s)}).' 
        
        s = np.repeat(self.col_labels, s) # Convert the sample to an array of ASVs where each ASV is repeated the number of times it appears in the sample.
        s = np.random.choice(s, n, replace=False) # Sample from the array of ASV labels. 

        if species_count_only:
            # For plotting rarefaction curves, it is much faster to return only the number of unique species.
            return len(np.unique(s))
        else:
            # This uses numpy broadcasting to generate an n-dimensional array of boolean values for each asv, indicating if it matches the element in sample. 
            # Basically converts sample from an array of ASV labels to an array of counts, with indices corresponding to self.col_labels.
            s = np.sum(self.col_labels[:, np.newaxis] == s, axis=1).ravel()
            return s
        


    
