'''Code for implementing various ordination approaches.'''
import numpy as np 
import pandas as pd 
from matrices import Matrix, AsvMatrix, CountMatrix
from typing import NoReturn


class CorrespondenceAnalysis():
    '''Class for performing Correspondence analysis on an AsvMatrix. Following procedure
    Described here: https://www.mathematica-journal.com/2010/09/20/an-introduction-to-correspondence-analysis/'''

    def __init__(self, n_components:int=2):

        self.row_scores = None
        self.col_scores = None
        self.row_labels = None
        self.col_labels = None
        self.n_components = n_components
        self.var_name = 'category'

    def groupby(self, matrix:CountMatrix, var:np.array):

        df = matrix.to_df()
        df[self.var_name] = var
        df = df.groupby(self.var_name, as_index=True).sum()

        self.col_labels = df.columns.values
        self.row_labels = df.index.values

        return df.values

    def fit(self, matrix:CountMatrix, var:np.array, var_name:str=None):

        self.var_name = var_name if (var_name is not None) else self.var_name
        matrix = self.groupby(matrix, var, var_name)

        P = matrix / np.sum(matrix) # Get the correspondence matrix by dividing entries by the grand total. 
        nrows, ncols = P.shape # Store number of rows and columns. 

        # Define row and column vector totals. 
        pi, pj = np.sum(P, axis=1, keepdims=True), np.sum(P, axis=0, keepdims=True)
        mu = pi * pj # Should have the same shape as the distance matrix. 
        # Calculate the matrix of standardized residuals residuals. 
        omega = P - mu
        omega = omega / np.sqrt(mu)
        # Apply singular value decomposition to the omega matrix. 
        V, lam, W = np.linalg.svd(omega, compute_uv=True)
        # Compute a diagonal matrix, where each entry is the reciprocal of the square root of the row totals. 
        delta_r = np.diag(1 / np.sqrt(pi))
        # Compute a diagonal matrix, where each entry is the reciprocal of the square root of the column totals. 
        delta_c = np.diag(1 / np.sqrt(pj))     

        self.row_scores = delta_r * V * np.diag(lam)
        self.column_scores = delta_c * W.T


class NonmetricMultidimensionalScaling():

    def __init__(self, n_components:int=2):
        pass

    def fit(self, matrix:Matrix):
        pass

    
    # def nmds(self, n_dims:int=2):
    #     '''Use non-metric multidimensional scaling to reduce the dimensions of the distance matrix
    #     and extract key data features.'''
        
    #     # Initialize the NMDS model, making sure to specify that the dissimilarities are precomputed. 
    #     model = MDS(n_components=2, metric=False, dissimilarity='precomputed')
    #     embeddings = model.fit_transform(self.matrix)