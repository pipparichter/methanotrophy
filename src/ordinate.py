'''Code for implementing various ordination approaches.'''
import numpy as np 
import pandas as pd 
from matrix import *
from typing import NoReturn
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
import umap
import pygam
from pygam import s
from transform import *
import copy
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy


class Ordination():

    def __init__(self, n_components:int=2):
        self.n_components = n_components
        self.row_scores = None
        self.col_scores = None # Not all ordination methods generate column scores!

        self.surface_models = {}
        self.vector_models = {}
        self.factor_models = {}

        # Some attributes for plotting. 
        self.xlabel, self.ylabel = None, None

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

    def plot(self, 
        title:str=None, 
        colors:pd.Series=None, 
        show_fit_vector:str=None,
        show_fit_surface:str=None) -> NoReturn:
        '''Plot ordinated points.

        :param nmds: A NonmetricMultiDimensionalScaling object which has been fitted to a CountMatrix. 
        :param labels: A pandas Series containing labels for each scatter point. 
        :param title: A title for the plot. 
        '''
        fig, ax = plt.subplots()
        
        if not pd.api.types.is_numeric_dtype(colors): # If the colors argument is a category...
            colors = pd.Categorical(colors).codes
        ax.scatter(self.row_scores[:, 0], self.row_scores[:, 1], c=colors)
        
        if show_fit_surface is not None:
            n = 100 # The number of points on each axis for which to generate values. 
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            x, y = np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n)
            # meshgrid produces two n by n arrays. Each element in the first array 
            xx, yy = np.meshgrid(x, y)

            surface_model = self.surface_models[show_fit_surface]
            # Input to model.predict should be of dimensions n_samples, n_features.
            z = surface_model.predict(np.vstack([xx.ravel(), yy.ravel()]).T)
            contour = ax.contour(xx, yy, z.reshape(xx.shape), colors='gray')
            ax.clabel(contour, inline=True, fontsize=10)

        ax.set_title('' if title is None else title)

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)


class PrincipalCoordinatesAnalysis(Ordination):

    def __init__(self, n_components=2, transform:bool=False, metric:str='bray-curtis'):
        
        super().__init__(n_components=n_components)
        self.transform = transform
        self.metric = metric

        self.eigenvalues = None

    def fit(self, M:CountMatrix):

        D = self._pre_pcoa(M) # Compute the distance matrix using the pre-specified metric.

        A = -0.5 * np.square(D)
        n = len(D)
        ones = np.ones(n)[np.newaxis].T
        I = np.eye(n)
        B = (I - ones.dot(ones.T) / n).dot(A).dot(I - ones.dot(ones.T) / n)

        eigenvalues, U = np.linalg.eig(B)

        # Indices for sorting the eigenvalues in descending order. 
        sort_idxs = eigenvalues.argsort()[::-1]
        U = U.real[:,sort_idxs]  # Sort the eigenvectors.

        # Sort the eigenvalues and store in the object. 
        self.eigenvalues = np.round(eigenvalues.real[sort_idxs], 4)
        self.row_scores = np.round(U.dot(np.diag(np.sqrt(self.eigenvalues))), 4)


    def _pre_pcoa(self, M):

        D = DistanceMatrix(metric=self.metric).from_count_matrix(M)
        assert np.all(D.matrix >= 0), 'NonmetricMultidimensionalScaling._pre_nmds: All values in the DistanceMatrix must be non-negative.'
        return D.to_numpy()



class CorrespondenceAnalysis(Ordination):
    '''Class for performing Correspondence analysis on a CountMatrix. Implementation follows the procedure
    described here: https://www.mathematica-journal.com/2010/09/20/an-introduction-to-correspondence-analysis/'''

    def __init__(self, n_components:int=2):
        '''Initialize a CorrespondenceAnalysis object.'''

        super().__init__(n_components=n_components)
        
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


class UMAP(Ordination):
    '''A class for using UMAP to reduce the dimensions of a CountMatrix object.'''
    # Map to convert any names to something recognized by the umap.UMAP class. 
    valid_metrics = {'bray-curtis':'braycurtis', 'euclidean':'euclidean'}

    def __init__(self, n_components:int=2, metric:str='bray-curtis'):
        # Should I transform the data for UMAP as well?
        super().__init__(n_components)

        assert metric in UMAP.valid_metrics
        self.model = umap.UMAP(n_components=n_components, metric=UMAP.valid_metrics[metric])
        
        # Some attributes for plotting.
        self.xlabel, self.ylabel = 'UMAP 1', 'UMAP 2'

    def fit(self, M:CountMatrix):
        '''Fit the CountMatrix to the model.'''

        self.row_scores = self.model.fit_transform(M.to_numpy())

# NOTE: I think the Procrustes analysis prevents the overfitting behavior I am seeing with high numbers of iterations. 
# Probably worth implementing the terination condition from metaMDS properly. 

class NonmetricMultidimensionalScaling(Ordination):
    '''A class for handling the non-metric multidimensional scaling of a CountMatrix object. The pre- and post-processing 
    options provided by this class, as well as the default parameters, are modeled on the metaMDS function in R's vegan package.'''
    def __init__(self, 
        n_components:int=2, 
        metric:str='bray-curtis', 
        transform:str=True, 
        max_iters:int=200,
        max_tries:int=20,
        half_change_scaling:bool=True,
        rotate_to_principal_components:bool=True):
        '''Initializes a NonmetricMultidimensionalScaling object.

        :param n_components: The number of dimensions to which the data will be reduced.
        :param metric: The dissimilarity measure to use to generate the distance matrix.
        :param transform: The transformation to apply to the CountMatrix prior to ordinating.
        '''
        super().__init__(n_components)

        assert metric in DistanceMatrix.valid_metrics, 'NonmetricMultidimensionalScaling: Dissimilarity metric is invalid.'
        self.metric = metric
        self.transform = transform
        
        # Keyword arguments to be used to instantiate the ScikitLearn NMDS model. 
        # self.

        self.max_tries, self.max_iters = max_tries, max_iters
        self.epsilon = 0.05
        self.residual_threshold = 0.01
        self.rmse_threshold = 0.005
        self.stress = None # Will store the stress of the final NMDS fit. 

        # Setting used during post-processing on NMDS solution. 
        self.half_change_scaling = half_change_scaling
        self.rotate_to_principal_components = rotate_to_principal_components


    def fit(self, M:CountMatrix, verbose:bool=True):
        '''Fit the data stored in a CountMatrix to the NMDS model.'''
        D = self._pre_nmds(M).to_numpy() # Compute the distance matrix and apply transformations, if specified. 

        # Not initializing with a metric MDS solution, as is done in metaMDS. Will see if that's a problem. 
        stress_best, sol_best = 1, None

        for i in range(self.max_tries):

            stress, sol, n_iters = self._nmds_ecopy(D, i=i)
            if verbose: print(f'Run {i} finished in {n_iters} iterations. Stress: {stress}')

            # Check to see if the stress of the new solution is less than the best stress value, or within a certain epsilon range.
            if (stress_best - stress) > -self.epsilon:
                proc_rmse, proc_residuals = NonmetricMultidimensionalScaling._procrustes(sol_best, sol) # Run Procrustes. 
                # Update the best solution if the stress is lower. 
                if stress < stress_best:
                    if verbose and (i > 1): print(f'New best solution found. Procrustes RMSE: {proc_rmse} Procrustes max residual {max(proc_residuals)}.')
                    stress_best, sol_best = stress, sol
                if (proc_rmse < self.rmse_threshold) and (max(proc_residuals) < self.residual_threshold):
                    # Something else might happen here in metaMDS -- termination criterion. 
                    if verbose and (i > 1): print(f'New best solution is similar to previous best.')
                    break

        # Store the NMDS results.
        self.stress = stress_best
        self.row_scores = sol_best
        self._post_nmds()

    def _nmds_sklearn(self, D:np.ndarray, i:int=None):
        '''An iterative NMDS algorithm using the ScikitLearn's implementation of NMDS.'''
        nmds_kwargs = {'eps':1e-4, 'n_components':self.n_components, 'max_iter':self.max_iters, 'n_init':1, 'normalized_stress':True, 'metric':False, 'dissimilarity':'precomputed'}
        nmds = MDS(**nmds_kwargs) # Initialize the ScikitLearn MDS object.
        nmds.fit(D)
        # Extract the relevant things from the fitted NMDS model. 
        n_iters = nmds.n_iter_
        sol = nmds.embedding_
        stress = nmds.stress_

        return stress, sol, n_iters

    def _nmds_ecopy(self, D:np.ndarray, i:int=None):
        '''An iterative NMDS algorithm mimicing the procedure in the ecoPy library. Due to broken dependencies, 
        this could not be used directly.'''

        if i == 0:  # If this is the first attempt at NMDS, try to initialize the algorithm with the PCoA solution. 
            pcoa = PrincipalCoordinatesAnalysis(n_components=self.n_components)
            pcoa.fit(self.M) # Fit using the CountMatrix stored in the object during _pre_nmds.
            sol = pcoa.row_scores
        else: # If this is not the first attempt, initialize with random numbers sampled from a uniform distribution.
            sol = np.random.rand(len(D), self.n_components)

        def _get_Vp():
            '''Not completely sure what this function is doing.'''
            n = len(D)
            Vp = np.ones((n, n)) # Initialize Vp as an n-by-n array of ones. 
            np.fill_diagonal(Vp, n)
            return np.linalg.pinv(Vp) # Calculate the pseudo-inverse. 
        
        def _get_stress(D_sol, D_hat):
            '''Compute the stress between the fitted dissimilarity matrix D and the Euclidean distance
            matrix of the proposed NMDS solution.'''
            return np.sqrt(((D_sol - D_hat) ** 2).sum() / (D_sol **2).sum())

        def _get_B(D_hat, D_sol):
            '''An adaptation of the Bcalc function in ecoPy. Modified to remove the multiplication by a weights
            matrix and skipping the step to convert values 1e-5 to 0 (I am not sure why this is done).'''
            B = -D_hat / D_sol # Compute the ratio of D_hat to D_sol. 
            np.fill_diagonal(B, np.abs(B).sum(axis=1))
            return B

        def _nmds_iter(D, sol, Vp):
            '''Carry out one iteration of the iterative NMDS algorithm.'''
            D_sol = DistanceMatrix(metric='euclidean').from_numpy(sol).to_numpy() # Get the Euclidean distance matrix of the ordination points. 
            D_sol[D_sol==0] = 1E-5 # Fill in zero values in the Euclidean distance matrix. 
            # Use isotonic regression to fit the dissimilarity matrix to D_sol.
            D_hat = IsotonicRegression().fit_transform(D.ravel(), D_sol.ravel()).reshape(D.shape)
            # Compute the stress between the Euclidean distance matrix from the embedded points and the isotonic regression
            # of the dissimilarity matrix to the Euclidean distance matrix of the embedded points. 
            stress = _get_stress(D_sol, D_hat)
            B = _get_B(D_hat, D_sol)
            Xu = Vp.dot(B).dot(sol)
            return stress, Xu

        Vp = _get_Vp()
        n_iters, delta, stress = 0, 1e4, 1
        while (n_iters < self.max_iters) and (delta > 1e-4):
            new_stress, Xu = _nmds_iter(D, sol, Vp)
            delta = stress - new_stress
            sol = PCA(n_components=self.n_components).fit_transform(Xu)
            stress = new_stress
            n_iters += 1    

        sol = NonmetricMultidimensionalScaling._scale_unit_rms(sol)
        return stress, sol, n_iters

    # @staticmethod
    def _scale_unit_rms(sol):
        '''Normalizes the row scores such so that the centroid is moved to the zero origin and the sum of squares
        is adjusted such that the root-mean-squared distance between points and the origin is 1.'''
        centers = np.mean(sol, axis=0)
        # assert len(centers) == self.n_components, 'NonmetricMultidimensionalScaling: The number of centers should be equal to the number of components.'
        sol = sol - centers

        # Calculate the RMS distance of centered points from the origin. 
        scaling_factor = len(sol) / np.sum(np.square(sol))
        scaling_factor = np.sqrt(scaling_factor)
        return sol * scaling_factor

    # Why does this function look different than the Python implementation?
    @staticmethod
    def _procrustes(sol_best, sol, scale:bool=True):
        '''Perform symmetric Procrustes analysis on two NMDS solutions. This function should mimic vegan's
        procrustes function, source code here: https://github.com/vegandevs/vegan/blob/master/R/procrustes.R
        
        :param scale: If True... ?
        '''
        if sol_best is None:
            return np.inf, np.inf

        # Setting translate to True centers the solutions, and scale normalizes by the Frobenius norm. 
        # Both of these data modifications are used in the vegan implementation of Procrustes. 
        # result = procrustes.symmetric.symmetric(sol_a, sol_b, pad=False, scale=True, tranlate=True)
        
        # Center each input, and scale by the Frobenius norm. 
        sol_best = sol_best - np.mean(sol_best, axis=0, keepdims=True)
        sol_best = sol_best / np.linalg.norm(sol_best)
        sol = sol - np.mean(sol, axis=0, keepdims=True)
        sol = sol / np.linalg.norm(sol)

        # cross_product = np.cross(sol_best, sol) # Does order matter here?
        
        # Returns the left singular vectors, the singular values, and the right singular vectors.
        # In R, the singular values are the "d" attribute of the SVD solution. 
        U, s, V_T = scipy.linalg.svd(sol_best.T @ sol) # Compute SVD of sol_a.
        A = np.matmul(V_T, U.T) 

        c = 1
        if scale:
            c = np.sum(s) / np.sum(sol**2)
        # Transform and scale the sol configuration. 
        sol_trans = c * np.matmul(sol, A)

        residuals = np.abs(sol_trans - sol_best).ravel()
        n = len(np.ravel(sol_trans))
        rmse = np.sqrt(np.sum((sol_trans - sol_best)**2) / n)

        return rmse, residuals
        # return {'rmse':rmse, 'residuals':residuals}
    
    def _post_nmds(self):
        '''Post-processing steps applied to the ordination scores produced by NMDS analysis.'''
        # If specified, center the ordination scores and rotate them to principal components. 
        if self._rotate_to_principal_components:
            self._rotate_to_principal_components()
        if self.half_change_scaling:
            self._scale_half_change()

    def _pre_nmds(self, M):
        '''This function mimics the matrix preprocessing performed in the metaMDSdist function. It applies some transformations to the
        data, depending on the value in the transform attribute and characteristics of the dataset. It also computes the distance matrix
        from the CountMatrix.'''
        # In the metaMDSdist function, both transformations can be applied (this is NOT supposed to be elif). 
        if (max(M) > 50) and self.transform:
            print('Square root transformation applied.')
            M = SquareRootTransformation()(M, inplace=False)
        if (max(M) > 9) and self.transform:
            print('Wisconsin double standardization applied.')
            M = WisconsinTransformation()(M, inplace=False)

        D = DistanceMatrix(metric=self.metric).from_count_matrix(M)
        # Store the matrices in the object. 
        self.D = D
        self.M = M

        assert np.all(D.matrix >= 0), 'NonmetricMultidimensionalScaling._pre_nmds: All values in the DistanceMatrix must be non-negative.'
        return D

    def _rotate_to_principal_components(self):
        '''Center the ordination scores and project on to principal components.'''
        # postMDS centers the scores using the scale function, with scale=FALSE. See https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/scale. 
        # This is just subtracting the column means.
        # The ScikitLearn implementation of PCA centers the data already (based on looking at source code), so not sure if this is necessary.
        centers = np.mean(self.row_scores, axis=0)
        assert len(centers) == self.n_components, 'NonmetricMultidimensionalScaling: The number of centers should be equal to the number of components.'
        self.row_scores = self.row_scores - centers
        # This tends to fail because of numerical errors. End up with something times like 10 to the -16.
        # assert np.sum(np.mean(self.row_scores, axis=0)) == 0, 'NonmetricMultidimensionalScaling: Centering of ordination scores failed.' 

        pca = PCA(n_components=None)
        self.row_scores = pca.fit_transform(self.row_scores)

    def _scale_half_change(self, threshold:float=0.8, n_threshold:int=10):
        '''Scales the scores so that one unit coresponds to a halving of community similarity from "replicate similarity." Adapted
        from the R source code for postMDS, found here  https://github.com/vegandevs/vegan/blob/master/R/postMDS.R.'''

        # Following the procedure from the vegan source code... https://github.com/vegandevs/vegan/blob/master/R/postMDS.R 
        assert self.row_scores is not None, 'NonmetricMultidimensionalScaling._scale_half_change: Row scores have not yet been computed.'
        D_sol = DistanceMatrix(metric='euclidean').from_numpy(self.row_scores).to_numpy() # Compute a matrix of Euclidean distances between the NMDS embeddings.
        D = self.D.to_numpy()

        # The dimensions of D_nmds and the dissimilarity matrix should match.
        assert D_sol.shape == D.shape, 'NonmetricMultidimensionalScaling._scale_half_change: Dimensions of the dissimilarity matrix and D_nmds do not match.'

        D_max = np.max(D) # Get the maximum dissimilarity value. 
        threshold = D_max * threshold # Adjust the threshold accordining to the maximum dissimilarity score. 
        idxs_below_threshold = D < threshold
        if np.sum(idxs_below_threshold) >= n_threshold:
            # Take the dissimilarity matrix D to be the outcome (or response) variable, and the Euclidean distance
            # matrix from the ordination scores D_row_scores to be the predictor variable. 
            linreg = LinearRegression(fit_intercept=True) # Need the intercept, which is taken to be the "replicate dissimilarity."
            # First input to fit is the X (predictor), second is y (response).
            # D[idxs] and D_row_scores[idxs] should both be one-dimensional arrays. 
            linreg.fit(D_sol[idxs_below_threshold].reshape(-1, 1), D[idxs_below_threshold].reshape(-1, 1))
            # When creating a linear regression model using a single predictor, the regression coefficient represents
            # the difference in the predicted value of the outcome variable for each one-unit increase in the predictor variable. 
            coefs = linreg.coef_ # Extract the coefficients of the fitted linear model. 
            scaling_factor = ((1 - linreg.intercept_) / 2) / coefs[0]
            # Apply the scaling factor to the ordination scores. 
            self.row_scores = self.row_scores / scaling_factor
        else:
            print('NonmetricMultidimensionalScaling._scale_half_change: Not enough points fell below the dissimilarity threshold. Half-change scaling cannot be used.')

    # # TODO: Honestly not quite sure what this is doing. Might want to try to figure it out a bit. 
    # def _scale(self) -> NoReturn:
    #     '''Scale the ordination points so that the Euclidean distance between ordination points corresponds
    #     more closely to the dissimilarity scores in the matrix used for NMDS.'''
    #     # Following the procedure from the vegan source code... https://github.com/vegandevs/vegan/blob/master/R/postMDS.R 
    #     assert self.row_scores is not None, 'NonmetricMultidimensionalScaling._scale: Row scores have not yet been computed.'
    #     # Compute a matrix of Euclidean distances between the NMDS embeddings. 
    #     D_nmds = DistanceMatrix(metric='euclidean').from_numpy(self.row_scores)

    #     # The dimensions of D_nmds and the dissimilarity matrix should match.
    #     assert D_nmds.shape() == self.D.shape, 'NonmetricMultidimensionalScaling._scale: Dimensions of the dissimilarity matrix and D_nmds do not match.'

    #     D_max = np.max(self.D) # Get the maximum dissimilarity value. 
    #     D_nmds_max = np.max(D_nmds.to_numpy())
    #     scaling_factor = D_nmds_max / D_max
    #     # Apply the scaling factor to the ordination scores. 
    #     self.row_scores = self.row_scores / scaling_factor
    #     self.scale = True
    


