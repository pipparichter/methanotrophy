import numpy as np
import pandas as pd
import itertools
from typing import NoReturn
import os


class Matrix():

    def __init__(self):

        self.matrix = None
        self.shape = None

    def __getitem__(self, i):
        return self.matrix[i]        



class DistanceMatrix(Matrix):

    def __init__(self, df:pd.DataFrame):
        '''Initialize a DistanceMatrix object.'''
        super().__init__() # Initialize the parent class.
        
        self.idxs = df.index
        self.matrix = df.values
        self.shape = self.matrix.shape


class AsvMatrix(Matrix):
    '''An object for working with ASV tables, which are matrices of counts where each row corresponds
    to a sample and each column is an ASV group.'''

    def __init__(self, df:pd.DataFrame=None, name:str=None):
        '''Initialize an AsvMatrix object.
        
        :param df: A pandas DataFrame containing, at minimum, columns serial_code (the sample ID), count, and asv.
        '''
        # Can we assume that all ASVs are present in every sample, but with a count of zero? I think yes, by looking
        # at the data file, but I should probably add an explicit check. 

        super().__init__() # Initialize the parent class. 
        
        if df is not None:
            df = df[['serial_code', 'asv', 'count']]
            df = df.groupby(by=['serial_code', 'asv']).sum()
            df = df.reset_index() # Converts the multi-level index to categorical columns. 
            df = df.pivot(columns=['asv'], index=['serial_code'], values=['count'])

            self.matrix = df.values
            self.asvs = df.columns.get_level_values('asv').values
            self.samples = df.index.values
            self.shape = self.matrix.shape
            self.normalized = False
            self.name = name

    def _sample(self, i:int, n:int, species_count_only:bool=False):

        s = self.matrix[i] # Get the sample from the matrix. 
        
        # It doesn't make sense to sample from an array of relative abundances, I think. 
        assert self.normalized == False, 'matrix.AsvMatrix.sample: The AsvMatrix has already been normalized.'
        assert n <= np.sum(s), f'matrix.AsvMatrix.sample: The sample size must be no greater than the total number of observations ({np.sum(s)}).' 
        
        s = np.repeat(self.asvs, s) # Convert the sample to an array of ASVs where each ASV is repeated the number of times it appears in the sample.
        s = np.random.choice(s, n, replace=False) # Sample from the array of ASV labels. 

        if species_count_only:
            # For plotting rarefaction curves, it is much faster to return only the number of unique species.
            return len(np.unique(s))
        else:
            # This uses numpy broadcasting to generate an n-dimensional array of boolean values for each asv, indicating if it matches the element in sample. 
            # Basically converts sample from an array of ASV labels to an array of counts, with indices corresponding to self.asvs.
            s = np.sum(self.asvs[:, np.newaxis] == s, axis=1).ravel()
            return s

    def normalize(self, method='rarefication'):
        '''Normalize the AsvMatrix using the specified approach.

        :param method: The normalization method to apply, one of 'rarefication' or 'proportion.'
        '''
        assert method in ['rarefication', 'proportion'], f'matrix.AsvMatrix.normalize: The normalization method {method} is not supported.'
        self.normalized = True
        
        if method == 'rarefication':
            self.normalize_rarefication()
        elif method == 'proportion':
            self.normalize_proportion()


    def normalize_rarefication(self):
        '''Normalize the AsvMatrix using rarefaction.'''
        n = min(np.sum(self.matrix, axis=1))
        self.matrix = np.apply_along_axis(lambda s : self._sample(s, n), 1, self.matrix)

    def normalize_proportion(self):
        pass

    def get_bray_curtis_distance_matrix(self):
        '''Calculate the Bray-Curtis similarity score for each pair of samples in the
        AsvMatrix object.'''

        # def bray_curtis_distance(i:int, j:int):
        #     '''Calculate the Bray-Curtis similarity score for samples at indices i and j.'''
        #     cij = np.sum([min(x, y) for x, y in zip(self.matrix[i], self.matrix[j])])
        #     si = np.sum(self.matrix[i]) # Total number of specimens counted on site i (or 1 if relative abundances are used).
        #     sj = np.sum(self.matrix[j]) # Total number of specimens counted on site j (or 1 if relative abundances are used).
        #     return 1 - (2 * cij) / (si + sj)

        # matrix = np.zeros((len(self.samples), len(self.samples)))
        # for i in range(len(self.samples)):
        #     for j in range(i, len(self.samples)):
        #         # The metric should be symmetric, so don't need to calculate twice. 
        #         bc = bray_curtis_distance(i, j)
        #         matrix[i, j] = bc
        #         matrix[j, i] = bc

        # Vectorizing sped this up from ~2 minutes to ~4 seconds!
        m = np.zeros((len(self.samples), len(self.samples))) # Initialize an empty distance matrix. 
        s = self.matrix.sum(axis=1) # Compute the total counts in each sample. 
        for i in range(len(self.samples)):
            x = self.matrix[i] # Grab the primary sample.
            # If axis = 1, apply_along_axis will loop through each row and implement the function to the row. 
            c = np.apply_along_axis(lambda y : np.where(np.less(y, x), y, x), 1, self.matrix).sum(axis=1)
            m[i] = 1 - (2 * c) / (s + s[i])

        df = pd.DataFrame(m, columns=self.samples, index=self.samples)
        return DistanceMatrix(df)
    
    def save(self, path:str) -> NoReturn:
        '''Save the matrix to a file.'''
        df = pd.DataFrame(self.matrix, columns=self.asvs, index=self.samples)
        df.to_cv(path)

    def load(self, path:str) -> NoReturn:
        '''Load the matrix from a file.'''
        df = pd.read_csv(path, index_col=0)
        self.__init__(df, name=os.path.basename(path))

