import numpy as np
import pandas as pd
from matrix import CountMatrix

# TODO: Might want to reevaluate how these are structured. Does it make more sense to write them as functions rather than
# classes? Should I separate normalizers versus transformers?
# Seems like scaling and normalization both fall under the umbrella of transformation. 
# https://medium.com/@isalindgren313/transformations-scaling-and-normalization-420b2be12300


class Transform():
    
    def __call(self, M:CountMatrix, inplace:bool=False) -> CountMatrix:
        pass


class Rarefaction(Transform):
    
    def __init__(self):
        super().__init__()

    def __call__(self, M:CountMatrix, n:int=None, inplace:bool=False) -> CountMatrix:
        '''Normalize the read depth of samples in the input CountMatrix using rarefaction.'''
        if not inplace: # If you don't want to modify the original CountMatrix, make sure to copy it. 
            M = M.copy()
        # It's recommended to choose the largest sample size possible when rarefying. 
        n = min(np.sum(M.matrix, axis=1)) if n is None else n
        M.matrix = np.array([self._sample(i, n) for i in range(len(self.matrix))])
        return M


class ConstantSumScaling(Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, M:CountMatrix, inplace:bool=False) -> CountMatrix:
        '''Normalize the read depth of samples by dividing by the total library size in each sample.'''
        if not inplace: # If you don't want to modify the original CountMatrix, make sure to copy it. 
            M = M.copy()
        norm_factor = M.matrix.sum(axis=1, keepdims=True) # Get the constant-sum normalization factor. 
        M.matrix = M.matrix / norm_factor
        return M


class CenteredLogRatioTransformation(Transform):
    pass


class SquareRootTransformation(Transform):
    '''Implements a square-root transformation of a CountMatrix. The square-root transformation is to 
    take the square root of every element in a non-normalized CountMatrix.'''
    def __init__(self):
        super().__init__()

    def __call__(self, M:CountMatrix, inplace:bool=False) -> CountMatrix:
        '''Applies the square-root transformation to the input CountMatrix.'''
        if not inplace: # If you don't want to modify the original CountMatrix, make sure to copy it. 
            M = M.copy()
        # Take the square root of each element in the underlying matrix. 
        M.matrix = np.sqrt(M.matrix)
        return M

class WisconsinTransformation(Transform):
    '''Implements a Wisconsin transformation of a CountMatrix. The Wisconsin transform is to 
    divide each element by its column maximum and then divide it by the row total.'''

    def __init__(self):
        super().__init__()

    def __call__(self, M:CountMatrix, inplace:bool=False) -> CountMatrix:
        if not inplace:
            M = M.copy()
        
        col_maxes = np.max(M.matrix, axis=0, keepdims=True)
        row_totals = np.sum(M.matrix, axis=1, keepdims=True)

        M.matrix = (M.matrix / col_maxes) / row_totals
        return M


class AncomNormalization(Transform):
    pass