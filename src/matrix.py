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
import warnings
from scipy.spatial.distance import braycurtis
import copy


def _reformat_csv(path:str, level:str='asv') -> pd.DataFrame:
    '''Use if the data is in long format, where there is a column for ASVs, a column for serial codes (the sample identifiers), and
    a column for counts. This means that there are n_asvs * n_samples entries in the data file. 
    
    :param path: The path to the CSV file which needs to be reformatted.
    :param level: The taxonomical level on which to merge the counts. 
    :return: A pandas DataFrame containing a matrix where rows are samples and columns are taxonomical units (resolution specified by the
        level parameter).
    '''
    assert level in CountMatrix.valid_levels, f'_format_csv: Specified level {level} is invalid.'
    # Read in the pandas DataFrame with Dask, which was found to be faster than using chunking. 
    ddf = dask.dataframe.read_csv(path, usecols=[level, 'count', 'serial_code'], dtype={level:str, 'count':int, 'serial_code':int}, blocksize=25e6)
    ddf = ddf.groupby(by=['serial_code', level]).sum()
    ddf = ddf.reset_index() # Converts the multi-level index to categorical columns. 
    ddf = ddf.compute() # Not totally sure why compute needs to be called here. Converts the Dask DataFrame to a pandas DataFrame. 
    ddf = ddf.pivot_table(columns=level, index='serial_code', values='count')
    # Reset column labels, which were weird because of the multi-indexing. 
    ddf.columns = ddf.columns.get_level_values(level).values
    return ddf



class Matrix():

    def __init__(self) -> NoReturn:

        self.metadata = None
        self.matrix = None
        self.row_labels = None
        self.col_labels = None

        # Store a history of the operations which have been applied to the Matrix.
        self.history = []

    def __getitem__(self, i):
        return self.matrix[i]        

    def __len__(self):
        return len(self.matrix)

    def __iter__(self):
        '''Implement a custom iterator method, so that min and max functions can be used on Matrix instances.'''
        return iter(self.matrix.ravel())

    def from_pandas(self, df:pd.DataFrame):
        '''Initialize a Matrix object from a pandas DataFrame.'''
        self.matrix = df.values
        self.row_labels = df.index.values
        self.col_labels = df.columns.values
        return self # Return a reference to the object to make the pipeline smoother.

    def from_numpy(self, arr:np.array, row_labels:List[str]=None, col_labels:List[str]=None):
        '''Initialize a Matrix object from a Numpy array.'''
        self.matrix = arr
        n_rows, n_cols = arr.shape
        self.row_labels = np.arange(n_rows) if row_labels is None else row_labels
        self.col_labels = np.arange(n_cols) if row_labels is None else col_labels
        return self # Return a reference to the object to make the pipeline smoother.

    def to_pandas(self) -> pd.DataFrame:
        '''Convert a Matrix object to a pandas DataFrame.'''
        return pd.DataFrame(self.matrix, index=self.row_labels, columns=self.col_labels)

    def to_numpy(self) -> np.array:
        '''Convert a Matrix object to a numpy array.'''
        return self.matrix

    def to_csv(self, path:str) -> NoReturn:
        '''Write a Matrix and associated metadata to a CSV file.'''
        # Use dask to process the underlying data for speed. This approach is substantially faster than writing in chunks. 
        ddf = self.to_pandas() # Convert to a DataFrame. 
        ddf = dask.dataframe.from_pandas(ddf, npartitions=500)
        ddf.to_csv(path, index=True)

    def from_csv(self, path:str, verbose:bool=False):
        '''Convert the DataFrame containing the sample metadata and ASV counts into an AsvMatrix object.'''
        assert self.level is not None, 'CountMatrix.from_csv: A level must be specified prior to loading in a CSV file. '
        ti = time.perf_counter()
        # Reading using Dask was found to be much faster than reading in chunks. 
        ddf = dask.dataframe.read_csv(path, blocksize=25e6, index_col=0)
        ddf = ddf.compute() # Converts the Dask DataFrame to a pandas DataFrame. 
        self.from_pandas(ddf) # Load the data into the objet. 
        tf = time.perf_counter()
        if verbose: print(f'Matrix.from_csv: CSV file loaded into Matrix in {np.round(tf - ti, 2)} seconds.')
        return self

    def __eq__(self, M):
        '''Test for equality between two Matrix objects.'''
        if M.shape() == self.shape(): # Check if matrices are the same shape. 
            return np.all(self.matrix == M.matrix)
        else: # If Matrix objects are not the same shape, then return False. 
            return False

    def transpose(self): # Possibly make inplace an option for this. 
        '''Transpose the underlying data, and return a new Matrix object with the transposed data.'''

        col_labels, row_labels = self.row_labels, self.col_labels
        df = pd.DataFrame(self.matrix.T, index=row_labels, columns=col_labels)
        # Instantiate a new Matrix object with the transposed data, depending on the type
        if self.__class__.__name__ == 'DistanceMatrix':
            new_matrix = DistanceMatrix(metric=self.metric)
        elif self.__class__.__name__ == 'CountMatrix':
            new_matrix = CountMatrix(level=self.level)
        new_matrix.from_pandas(df) # Add the data to the new Matrix.
        new_matrix.metadata = self.metadata # Copy the metadata into the new matrix. 
        return new_matrix

    def shape(self) -> Tuple[int, int]:
        '''Returns the shape of the underlying matrix.'''
        return self.matrix.shape


class DistanceMatrix(Matrix):

    valid_metrics = ['bray-curtis', 'euclidean', 'chi-squared']

    def __init__(self, metric:str=None):
        '''Initialize a DistanceMatrix object.'''
        super().__init__() # Initialize the parent class.
        assert metric in DistanceMatrix.valid_metrics, f'DistanceMatrix.__init__: Metric {metric} is invalid.'
        self.metric = metric

    def copy(self):
        '''Copy the underlying object.'''
        new_matrix = DistanceMatrix(metric=self.metric)
        for k, v in self.__dict__.items():
            setattr(new_matrix, k, copy.deepcopy(v))
        return new_matrix

    @staticmethod
    def _compute_chi_squared_distances(arr:np.array) -> NoReturn:
        '''Create a distance matrix using chi-squared distance between rows in the input array. 
        Used formula from https://link.springer.com/referenceworkentry/10.1007/978-0-387-32833-1_53.'''
        P = array / np.sum(array) # Get the relative frequency table by dividing entries by the grand total. 
        nrows, ncols = P.shape # Store number of rows and columns. 
        # Define row and column vector totals. 
        pi, pj = np.sum(P, axis=1, keepdims=True), np.sum(P, axis=0, keepdims=True)
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
        return D # Store the distance matrix in the object. 

    @staticmethod
    def _compute_bray_curtis_distances(arr:np.array) -> NoReturn:
        '''Create a distance matrix using Bray-Curtis dissimilarity between rows in the input array.'''
        # Vectorizing sped this up from ~2 minutes to ~4 seconds!
        n_rows = len(arr)
        D = np.zeros((n_rows, n_rows)) # Initialize an empty distance matrix. 
        s = arr.sum(axis=1) # Compute the total counts in each sample. 
        for i in range(n_rows):
            x = arr[i] # Grab the primary sample.
            D[i] = np.apply_along_axis(lambda y: braycurtis(y, x), 1, arr)
        return D 

    @staticmethod
    def _compute_euclidean_distances(arr:np.array) -> NoReturn:
        '''Create a distance matrix using Euclidean distance between rows in the input array. The formula used here is 
        up on the Wikipedia page for Euclidean distance matrices.'''
        # n = len(arr)
        # D = np.sum(arr**2, axis=1).reshape(n, 1).repeat(n, axis=1) + np.sum(arr**2, axis=1).reshape(1, n).repeat(n, axis=0) 
        # D -= 2 * np.matmul(arr, arr.T)
        # D = np.sqrt(np.round(D, 10))
        # return D
        
        # Why is this not working?
        n_rows = len(arr)
        D = np.zeros((n_rows, n_rows)) # Initialize an empty distance matrix. 
        for i in range(n_rows):
            x = arr[i, :]
            D[i] = np.sum(np.square(x - arr), axis=1)
        return np.sqrt(D)

    def from_numpy(self, arr:np.array, row_labels:List[str]=None, col_labels:List[str]=None):
        '''Initialize the DistanceMatrix object from a Numpy array.'''
        if self.metric == 'bray-curtis':
            D = DistanceMatrix._compute_bray_curtis_distances(arr)
        elif self.metric == 'euclidean':
            D = DistanceMatrix._compute_euclidean_distances(arr)
        elif self.metric == 'chi-squared':
            D = DistanceMatrix._compute_chi_squared_distances(arr) 

        return super().from_numpy(D, row_labels=row_labels, col_labels=col_labels)

    def from_count_matrix(self, M:Matrix):
        '''Initialize a DistanceMatrix object from a CountMatrix.'''
        return self.from_numpy(M.matrix, row_labels=M.row_labels, col_labels=M.col_labels)


class CountMatrix(Matrix):
    '''A matrix which contains observation counts in each cell. '''
    valid_levels = ['phylum', 'class', 'order', 'genus', 'species', 'family', 'domain', 'asv']

    def __init__(self, level:str='asv'):
        '''Initialize a CountMatrix object.'''
        super().__init__() # Initialize the parent class.
        assert level in CountMatrix.valid_levels, 'CountMatrix.__init__: Level {level} is invalid.'
        self.level = level

    def copy(self):
        '''Copy the underlying object.'''
        new_matrix = CountMatrix(level=self.level)
        for k, v in self.__dict__.items():
            setattr(new_matrix, k, copy.deepcopy(v))
        return new_matrix

    def get_metadata(self, field:str) -> pd.Series:
        '''Extract information from a particular field in the metadata attribute.
        
        :param field: The name of the metadata field to extract.
        :return: A pandas Series containing the metadata values. 
        '''
        assert field in self.get_metadata_fields(), f'CountMatrix.get_metadata: There is no field {field} in the CountMatrix metadata.'
        # Extract the data from the metadata table.
        return self.metadata[field]

    def get_metadata_fields(self) -> List[str]:
        '''Return a list of the fields contained in the underlying metadata.'''
        assert self.metadata is not None, 'CountMatrix.get_metadata_fields: There is no metadata stored in the CountMatrix object.'
        return list(self.metadata.columns)

    def filter_empty_cols(self, verbose:bool=True):
        '''Remove empty columns from the CountMatrix. Empty columns can occur when a subset of the total
        data is loaded, and not all ASVs or Taxonomical categories are represented.'''
        non_empty_idxs = np.sum(self.matrix, axis=0) > 0
        num_empty_cols = len(self.col_labels) - np.sum(non_empty_idxs)
        if verbose: print(f'CountMatrix.filter_empty_cols: Removing {num_empty_cols} empty columns from the CountMatrix.')
        self.col_labels = self.col_labels[non_empty_idxs]
        self.matrix = self.matrix[:, non_empty_idxs]
        return self # Return a reference to the object.

    def filter_read_depth(self, min_depth:int):
        '''Filter the matrix so that only samples with more than the specified number of reads are kept.'''
        depths = np.sum(self.matrix, axis=1)
        idxs = depths >= min_depth
        print(f'matrices.CountMatrix.filter_read_depth: Discarding {len(self.matrix) - np.sum(idxs)} samples with read depth less than {min_depth}.')

        self.matrix = self.matrix[idxs]
        self.row_labels = self.row_labels[idxs]
        # If metadata is present, make sure to drop filtered samples. 
        if self.metadata is not None:
            self.metadata = self.metadata[self.metadata.serial_code.isin(self.row_labels)]
        return self

    def sample(self, i:int, n:int, species_count_only:bool=False):
        '''Returns either an integer indicating the number of unique species in the sample or an array.'''
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
        
    def load_metadata(self, path:str=None, index_col='serial_code', usecols=['site', 'season', 'flux_ch4', 'temp_air', 'temp_soil', 'water_content', 'bulk_density']):
        '''Read in the specified fields from a metadata file into a pandas DataFrame, and store as an attribute. If there is more than one
        entry for a given sample, then all duplicate rows after the first row are removed from the metadata.'''

        metadata = pd.read_csv(path, usecols=usecols, index_col=index_col) 
        ni = len(metadata)
        metadata = metadata[~metadata.index.duplicated(keep='first')]
        nf = len(metadata)
        print(f'Matrix.load_metadata: {ni - nf} duplicate rows were removed from the metadata.')
        self.metadata = metadata # Store the metadata in the object.

    
