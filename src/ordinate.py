'''Code for implementing various ordination approaches.'''
import numpy as np 
import pandas as pd 
from matrix import *
from typing import NoReturn
from sklearn.manifold import MDS
import pygam
from pygam import s


class Ordination():

    def __init__(self, n_components:int=2, scaling:int=1):
        self.n_components = n_components
        self.row_scores = None
        self.col_scores = None # Not all ordination methods generate column scores!

        self.surface_models = {}
        self.vector_models = {}
        self.factor_models = {}


    def fit(self, matrix:CountMatrix) -> NoReturn:
        '''This function should be overloaded in derivative classes.'''
        pass

    def fit_vector(self, y:pd.Series):
        # https://stats.stackexchange.com/questions/56427/vector-fit-interpretation-nmds 
        pass

    def fit_surface(self, y:pd.Series, k:int=10) -> NoReturn:
        '''Fits a surface to the ordination space, taking the input variable as the response variable.
        This function only supports modelling of the first two ordination axes. 
        
        :param y: A single pandas Series containing floats, with length equal to len(row_scores).
        :param k: A value indicating the complexity of the splines used for each component. This value should
            not exceed the length of the number of unique values present in the var array.  
        '''
        assert y.dtype in [float, np.float64], 'Ordination.fit_surface: The y array must contain measurements of a continuous variable.'
        assert self.row_scores is not None, 'Ordination.fit_surface: Ordination object has not yet been fitted.'
        assert k < len(set(y)), 'Ordination.fit_surface: Spline complexity is too high for the given data.'
        assert len(y) == len(self.row_scores), 'Ordination.fit_surface: The number of measurements of the environmental variable does not match the number of ordination scores.'
        
        # Define the formula for the GAM. Each component will be modeled by a smoothing spline with complexity k.
        formula = s(0, n_splines=k) + s(1, n_splines=k)  
        gam = pygam.pygam.GAM(formula, distribution='normal', link='identity')
        
        # Expect the row scores to be of shape n by m >= 2, where n is the number of observations. 
        X = self.row_scores[:, :2] # Grab the first two components of the row scores. 
        gam.fit(X, y.values) # Fit the GAM to the data. 

        self.surface_models[y.name] = gam

    def fit_factor(self, y:pd.Series):
        pass


class CorrespondenceAnalysis(Ordination):
    '''Class for performing Correspondence analysis on a CountMatrix. Implementation follows the procedure
    described here: https://www.mathematica-journal.com/2010/09/20/an-introduction-to-correspondence-analysis/'''

    def __init__(self, n_components:int=2, scaling:int=1):
        '''Initialize a CorrespondenceAnalysis object.'''

        super().__init__(n_components=n_components, scaling=scaling)
        
        self.pi, self.pj = None, None
        self.P = None

    def fit(self, matrix:CountMatrix, field:str, n_bins:int=3) -> NoReturn:
        '''Use correspondence analysis to ordinate the data contained in a CountMatrix.

        :param matrix: The CountMatrix containing the data to ordinate.
        :param field: The metadata field for which correspondence will be analyzed. 
        :param n_bins: The number of bins into which the field data will be sorted. This field 
            should be specified if the metadata associated with the given field is not already categorical.
        '''
        P = matrix.matrix / np.sum(matrix.matrix) # Get the frequency matrix by dividing entries by the grand total. 
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
        self.row_scores = np.matmul(np.matmul(delta_r, V), np.diag(lam))
        self.col_scores = np.matmul(delta_c, W.T)

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

        assert metric in ['bray-curtis', 'chi-squared'], 'NonmetricMultidimensionalScaling: Dissimilarity metric is invalid.'
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
        self.row_scores = model.fit_transform(matrix.matrix)

