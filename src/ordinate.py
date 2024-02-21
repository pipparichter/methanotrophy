'''Code for implementing various ordination approaches.'''
import numpy as np 
import pandas as pd 
from matrix import *
from typing import NoReturn
from sklearn.manifold import MDS


class Ordination():

    def __init__(self, n_components:int):
        self.n_components = n_components

    def fit(self, matrix:CountMatrix) -> NoReturn:
        pass


class CorrespondenceAnalysis(Ordination):
    '''Class for performing Correspondence analysis on a CountMatrix. Implementation follows the procedure
    described here: https://www.mathematica-journal.com/2010/09/20/an-introduction-to-correspondence-analysis/'''

    def __init__(self, n_components:int=2):
        '''Initialize a CorrespondenceAnalysis object.'''

        super().__init__(n_components)
        
        self.row_embeddings = None
        self.col_embeddings = None
        self.row_labels = None
        self.col_labels = None
        self.field = None

        self.pi, self.pj = None, None
        self.P = None

    def bin(self, data:np.array, n_bins:int=None):

        if n_bins is None: # If no bins are specified, just return the raw flux values for each sample. 
            return data
        # Get the bin boundaries. 
        bins = np.linspace(min(data), max(data) + 1.0, n_bins + 1)
        # Lower bin boundary is inclusive, upper boundary is not. 
        binned_data = np.digitize(data, bins, right=False)
        # All bins indices should be between 1 and n_bins.
        return binned_data, bins

    def groupby(self, matrix:CountMatrix, field:str, n_bins:int=None):

        data = matrix.get_metadata(field)
        data_binned = self.bin(data, n_bins=n_bins)
        
        df = matrix.to_df() # Convert the CountMatrix to a DataFrame. 
        df[field] = data_binned
        df = df.groupby(field, as_index=True).sum()

        self.col_labels = df.columns.values
        self.row_labels = df.index.values

        return df.values

    def fit(self, matrix:CountMatrix, field:str, n_bins:int=3) -> NoReturn:
        '''Use correspondence analysis to ordinate the data contained in a CountMatrix.

        :param matrix: The CountMatrix containing the data to ordinate.
        :param field: The metadata field for which correspondence will be analyzed. 
        :param n_bins: The number of bins into which the field data will be sorted. This field 
            should be specified if the metadata associated with the given field is not already categorical.
        '''
        assert len(var) == len(matrix), 'ordination.CorrespondenceAnalysis.__call__: Length of the categorical variable array should be equal to the length of the CountMatrix.'
        self.field = field # Store the name of the variable used for grouping. 

        matrix = self.groupby(matrix, field, n_bins=n_bins)

        P = matrix / np.sum(matrix) # Get the frequency matrix by dividing entries by the grand total. 
        nrows, ncols = P.shape # Store number of rows and columns. 

        # Define row and column vector totals. 
        pi, pj = np.sum(P, axis=1, keepdims=True), np.sum(P, axis=0, keepdims=True)
        self.pi, self.pj, self.P = pi, pj, P # Store some intermediates in the object. 

        mu = pi * pj # Should have the same shape as the distance matrix. 
        # Calculate the matrix of standardized residuals residuals. 
        omega = P - mu
        omega = omega / np.sqrt(mu)
        # Apply singular value decomposition to the omega matrix. 
        V, lam, W = np.linalg.svd(omega, compute_uv=True)
        # Compute a diagonal matrix, where each entry is the reciprocal of the square root of the row totals. 
        delta_r = np.diag(1 / np.sqrt(pi.ravel()))
        # Compute a diagonal matrix, where each entry is the reciprocal of the square root of the column totals. 
        delta_c = np.diag(1 / np.sqrt(pj.ravel()))    
        self.row_embeddings = np.matmul(np.matmul(delta_r, V), np.diag(lam))
        self.col_embeddings = np.matmul(delta_c, W.T)

    def get_inertia(self):
        '''Caculate the inertia of the CA representation, which is a way to guage the quality of the
        representation. Formula is derived from https://www.mathematica-journal.com/2010/09/20/an-introduction-to-correspondence-analysis/'''
        # The centroid of the normalized matrix P is just the column totals. 
        centroid = self.pj

        d = np.square((self.P / self.pi) - centroid)
        d = d / centroid
        # Sum over the columns, which should be axis 1. 
        d = np.sum(d, axis=1)
        # d should now contain the chi-squared distance between each row and the centroid. 
        # d = np.sqrt(d) # Don't take the square root, as d is squared in the next step anyway. 

        # Calculate the total inertia of the rows. 
        inertia = np.sum(self.pi * d)
        return inertia


class NonmetricMultidimensionalScaling(Ordination):
    '''A class for handling the non-metric multidimensional scaling of a CountMatrix object.'''

    def __init__(self, n_components:int=2, metric:str='bray-curtis'):
        '''Initializes a NonmetricMultidimensionalScaling object.

        :param n_components: The number of dimensions to which the data will be reduced.'''
        super().__init__(n_components)

        assert metric in ['bray-curtis', 'chi_squared'], 'NonmetricMultidimensionalScaling: Dissimilarity metric is invalid.'
        self.metric = metric
        self.embeddings = None

        self.matrix = None # Where the computed distance matrix will be stored. 

    def fit(self, matrix:CountMatrix):
        
        if self.metric == 'bray-curtis':
            matrix = matrix.get_bray_curtis_distance_matrix()
        elif self.metric == 'chi-squared':
            matrix = matrix.get_chi_squared_distance_matrix()

        # Set metric=False for non-metric MDS. Set dissimilarity to precomputed, as distance matrix is calculated manually.
        # When should I use normed_stress? 
        model = MDS(n_components=self.n_components, metric=False, dissimilarity='precomputed', normalized_stress='auto')
        self.embeddings = model.fit_transform(matrix.matrix)

