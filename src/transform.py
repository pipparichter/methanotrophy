import numpy as np
import pandas as pd
from matrix import CountMatrix
import scipy


class Transform():

    def __call__(self, M:CountMatrix, inplace:bool=False, **kwargs) -> CountMatrix:

        if not inplace: # If you don't want to modify the original CountMatrix, make sure to copy it. 
            M = M.copy()
        # Call the _apply function specific to the derived class. 
        return type(self)._apply(M, **kwargs)

        @staticmethod
        def _apply(M:CountMatrix, **kwargs) -> CountMatrix:
            '''This method will be overloaded in derived classes.'''
            raise Exception('Transform: Transformation is not implemented.')


class Rarefaction(Transform):

    @staticmethod
    def _apply(self, M:CountMatrix, n:int=None) -> CountMatrix:
        '''Normalize the read depth of samples in the input CountMatrix using rarefaction.'''

        # It's recommended to choose the largest sample size possible when rarefying. 
        n = min(np.sum(M.matrix, axis=1)) if n is None else n
        M.matrix = np.array([self._sample(i, n) for i in range(len(self.matrix))])
        return M

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
        


class ConstantSumScaling(Transform):
    '''Implementation of constant-sum scaling, which is simply the division of each abundance observation by the total number
    of observations in the sample.'''

    @staticmethod
    def _apply(M:CountMatrix) -> CountMatrix:
        '''Normalize the read depth of samples by dividing by the total library size in each sample.'''

        norm_factor = M.matrix.sum(axis=1, keepdims=True) # Get the constant-sum normalization factor. 
        M.matrix = M.matrix / norm_factor
        return M


class CenteredLogRatioTransformation(Transform):
    '''Implementation of a centered log-ratio transformation, which is defined as log(x/g(x)), where g(x) is the geometric
    mean of all parts in the sample. This transformation requires the addition of
    a psudocount to handle zeros in the count matrix.
    
    Sources:
    (1) https://search.r-project.org/CRAN/refmans/compositions/html/clr.html 
    (2) https://www.geo.fu-berlin.de/en/v/soga-r/Advances-statistics/Feature-scales/Logratio_Transformations/index.html 
    
    '''
    @staticmethod
    def _apply(M:CountMatrix, psuedo_count:float=1.0) -> CountMatrix:
        '''Apply the CLR transformation to the input CountMatrix.
        
        :param pseudocount: A scalar value to add to each entry in the CountMatrix to avoid error caused by
            the presence of zeros.
        :return: The modified CountMatrix object. 
        '''
        # Add the pseudo count to the matrix. 
        m = M.matrix + psuedo_count
        # Compute the geometric means of each sample.
        geometric_means = scipy.stats.gmean(M.matrix, axis=1, keepdims=True)
        m = np.log(m / geometric_means)
        M.matrix = m # Store the transformed values in the CountMatrix. 
        return M


class SquareRootTransformation(Transform):
    '''Implementation of a square-root transformation. This has the effect of "stretching out"
    low-abundance values relative to high-abundance values.
    
    Sources:
    (1) https://www.youtube.com/watch?v=_uDv7LRUUsY&list=PLOPiWVjg6aTzsA53N19YqJQeZpSCH9QPc&index=12
    '''
    def _apply(self, M:CountMatrix) -> CountMatrix:
        '''Apply the square-root transformation to the input CountMatrix.'''
        # Take the square root of each element in the underlying matrix. 
        M.matrix = np.sqrt(M.matrix)
        return M


class WisconsinTransformation(Transform):
    '''Implements a Wisconsin transformation of a CountMatrix. The Wisconsin transform is to 
    divide each element by its column maximum and then divide it by the row total.'''

    def _apply(self, M:CountMatrix) -> CountMatrix:
        
        col_maxes = np.max(M.matrix, axis=0, keepdims=True)
        row_totals = np.sum(M.matrix, axis=1, keepdims=True)

        M.matrix = (M.matrix / col_maxes) / row_totals
        return M


class AncomNormalization(Transform):
    pass