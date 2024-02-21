import numpy as np
import pandas as pd
from matrix import CountMatrix

class Normalization():
    pass


class Rarefaction(Normalization):
    
    def __init__(self):
        super().__init__()

    def __call__(self, matrix:CountMatrix, n:int=None):
        '''Normalize the read depth of samples in the input CountMatrix using rarefaction.'''
        # It's recommended to choose the largest sample size possible when rarefying. 
        n = min(np.sum(matrix.matrix, axis=1)) if n is None else n
        matrix.matrix = np.array([self._sample(i, n) for i in range(len(self.matrix))])


class ConstantSumScaling(Normalization):
    def __init__(self):
        super().__init__()

    def __call__(self, matrix:CountMatrix):
        '''Normalize the read depth of samples by dividing by the total library size in each sample.'''
        norm_factor = matrix.matrix.sum(axis=1, keepdims=True) # Get the constant-sum normalization factor. 
        matrix.matrix = matrix.matrix / norm_factor


class CenteredLogRatioTransformation(Normalization):
    pass


class AncomNormalization(Normalization):
    pass