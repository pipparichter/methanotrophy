'''Code for implementing various ordination approaches.'''
import numpy as np 
import pandas as pd 
from matrix import *
from typing import NoReturn, Tuple, Dict
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
from collections import namedtuple

# Set all matplotlib global parameters.
plt.rc('font', **{'family':'sans-serif', 'sans-serif':['Arial'], 'size':10})
plt.rc('xtick', **{'labelsize':10})
plt.rc('ytick', **{'labelsize':10})
plt.rc('axes',  **{'titlesize':10, 'labelsize':10})
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set2.colors)
plt.rcParams['image.cmap'] = 'summer'
# plt.rcParams['image.cmap'] = 'gist_earth'

# TODO: Figure out if surface fit should be carried out before or after scaling. I think after, but not completely sure. 
# Because metadata is by sample, it makes sense to do the fit on the row scores only. Maybe throw a warning if scaling is 2?

class Ordination():

    VectorFit = namedtuple('VectorFit', ['r', 'coeffs', 'p'])
    FactorFit = namedtuple('FactorFit', ['r', 'coeffs', 'p'])
    SurfaceFit = namedtuple('SurfaceFit', ['gam'])

    def __init__(self, n_components:int=2, axes_labels:Tuple[str, str]=None):
        self.n_components = n_components
        self.row_scores = None # AKA site scores. 
        self.col_scores = None # AKA species scores. Not all ordination methods generate column scores!

        self.surface_fits = {}
        self.vector_fits = {}
        self.factor_fits = {}

        # Some attributes for plotting. 
        self.x_axis_label, self.y_axis_label = axes_labels

    def _prep(self, M:CountMatrix) -> Dict[str, object]:
        '''Pre-ordination procedure for putting the abundance data into an appropriate form for the particular
        ordination method.'''
        # If no particular postprocessing procedure is implemented in the derived class, just pass the unmodified 
        # abundance data into the fit method.
        return {'M':M}

    def _fit(self, **kwargs) -> Dict[str, object]:
        raise Exception('Ordination._fit: No _fit method has been implemented for this class.')

    def _post(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        '''Post-ordination procedure for generating final row scores.'''
        # If no particular postprocessing procedure is implemented, assume the inputs to the function contain
        # the completed row and column scores. 
        return kwargs.get('row_scores'), kwargs.get('column_scores')

    def fit(self, M:CountMatrix) -> NoReturn:
        '''Defines an ordination pipeline which can be implemented for each Ordination subclass. The pipeline handles
        preprocessing of the input abundance matrix, the ordination algorithm, and post-ordination modification of the resulting
        data.
        
        :param M: A CountMatrix containing the abundance data to use for ordination. 
        :return: Returns nothing, but stores ordination scores in the object.
        '''
        inputs = self._prep(M)
        outputs = self._fit(**inputs)
        self.row_scores, self.col_scores = self._post(**outputs)

    def fit_vector(self, y:pd.Series, n_permutations:int=999):
        '''This is analagous to vegan's vectorfit function, which is called by envfit. The function 
        fits environmental vectors onto an ordination, so that projection of points onto vectors have maximum 
        correlations with corresponding environmental variables.'''
        # https://stats.stackexchange.com/questions/56427/vector-fit-interpretation-nmds 
        name = y.name

        # Center the data prior to fitting the model. 
        X = self.row_scores - np.mean(self.row_scores, axis=0) # TODO center.
        y = y.values - np.mean(y.values)

        # The vegan implementation of this function uses QR decomposition to solve the least-squares regression. In Python, 
        # the lstq function is preferred: https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html
        coeffs, residuals, _, _  = np.linalg.lstsq(X, y, rcond=None) # X is being fit to Y. coeffs is the coefficients of the linear regression.
        X_fit = np.matmul(X, coeffs.reshape(self.n_components, 1)) # This projection should be one-dimensional. 
        # corrcoeff specifies that each row is a feature, not an observation. 
        r = scipy.stats.pearsonr(X_fit.ravel(), y.ravel()).statistic

        if n_permutations is not None:
            # Calculate a p-value for the fit by randomly permuting the values in y
            r_perms = [] # List to store the R value for all permutations. 
            for _ in range(n_permutations):
                np.random.shuffle(y) # Shuffle y to get a random permutation. 
                coeffs, residuals, _, _  = np.linalg.lstsq(X, y, rcond=None) # X is being fit to Y. coeffs is the coefficients of the linear regression. 
                X_fit = np.matmul(X, coeffs.reshape(self.n_components, 1)) # This projection should be one-dimensional. 
                r_perms.append(scipy.stats.pearsonr(X_fit.ravel(), y.ravel()).statistic)

            # To calculate the p-value for a permutation test, we simply count the number of test-statistics as or 
            # more extreme than our initial test statistic, and divide that number by the total number of test-statistics we calculated.
            p = (np.array(r_perms) >= r).astype(int).sum() / n_permutations
        else:
            p = None

        # Make the prediction a unit vector. 
        coeffs = coeffs / np.linalg.norm(coeffs)
        # Store the vector model. 
        self.vector_fits[name] = Ordination.VectorFit(r, coeffs, p)

    def fit_surface(self, y:pd.Series, k:int=10) -> NoReturn:
        '''Fits a surface to the ordination space, taking the input variable as the response variable.
        This function only supports modelling of the first two ordination axes. This should be analagous to the
        ordisurf function in vegan. 
        
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

        self.surface_fits[y.name] = Ordination.SurfaceFit(gam)

    def fit_factor(self, y:pd.Series):
        pass

    def plot(self, 
        title:str=None, 
        colors:pd.Series=None, 
        show_vector_fit:List[str]=None,
        show_surface_fit:str=None,
        biplot:bool=False) -> plt.axes:
        '''Plot ordinated points.

        :param nmds: A NonmetricMultiDimensionalScaling object which has been fitted to a CountMatrix. 
        :param labels: A pandas Series containing labels for each scatter point. 
        :colors: Values with which to color the points on the ordination plot. 
        :param title: A title for the plot. 
        '''
        fig, ax = plt.subplots()

        if colors is None:
            colors = pd.Series([0] * len(self.row_scores))

        # Check if the colors are numeric. If not, assume they are categories. 
        categorical = not pd.api.types.is_numeric_dtype(colors)
        if categorical: 
            s = ax.scatter(self.row_scores[:, 0], self.row_scores[:, 1], c=pd.Categorical(colors).codes)
            # Passing in np.unique(colors.values) should ensure that the order of the categories matches the colors.
            ax.legend(handles=s.legend_elements()[0], labels=list(np.unique(colors.values)))
        else:
            s = ax.scatter(self.row_scores[:, 0], self.row_scores[:, 1], c=colors.values)
        
        if show_surface_fit is not None:
            n = 100 # The number of points on each axis for which to generate values. 
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            x, y = np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n)
            # meshgrid produces two n by n arrays. Each element in the first array 
            xx, yy = np.meshgrid(x, y)

            gam = self.surface_fits[show_surface_fit].gam
            # Input to model.predict should be of dimensions n_samples, n_features.
            z = gam.predict(np.vstack([xx.ravel(), yy.ravel()]).T)
            contour = ax.contour(xx, yy, z.reshape(xx.shape), colors='gray')
            ax.clabel(contour, inline=True, fontsize=10)
        
        if show_vector_fit is not None:
            scale = ax.get_ylim()[1] # Scale the vector magnitude so it shows up on the plot. 
            for name in show_vector_fit:
                vf = self.vector_fits[name]
                x, y = vf.r * vf.coeffs * scale
                # Make sure this is right.
                ax.arrow(0, 0, x, y, width=0.1, head_width=0.5, color='black')
                ax.annotate(name, xy=(x, y), xytext=(x + 1, y + 1))

        ax.set_title('' if title is None else title)

        ax.set_xlabel(self.x_axis_label)
        ax.set_ylabel(self.y_axis_label)

        return ax

    
    def permutation_test(self, n:int=999):
        # Permutation test TODO
        pass


class PrincipalCoordinatesAnalysis(Ordination):

    def __init__(self, n_components=2, transform:bool=False, metric:str='bray-curtis'):
        
        super().__init__(n_components=n_components, axes_labels=('PCoA_1', 'PCoA_2'))
        self.transform = transform
        self.metric = metric

        self.eigenvalues = None

    
    def _pre(self, M:CountMatrix):
        '''Preprocessing for PrincipalCoordinatesAnalysis. Involves computing a distance matrix to be decomposed, using the metric
        specified upon instantiation.'''
        D = DistanceMatrix(metric=self.metric).from_count_matrix(M)
        assert np.all(D.matrix >= 0), 'NonmetricMultidimensionalScaling._pre_nmds: All values in the DistanceMatrix must be non-negative.'
        return {'D':D.to_numpy()}

    def _fit(self, D:np.ndarray) -> Dict[str, object]:
        '''Carry out PCoA on the input distance matrix D.

        param D: A distance matrix computed using the metric stored in self.metric.
        return: A dictionary containing the row and column scores generated by PCoA.'''
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
        # self.eigenvalues = np.round(eigenvalues.real[sort_idxs], 4)
        return {'row_scores':np.round(U.dot(np.diag(np.sqrt(self.eigenvalues))), 4), 'col_scores':None}

    def _post(self, row_scores:np.ndarray=None, col_scores:np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:
        '''Postprocessing for PCoA row scores.'''
        return row_scores, col_scores


class CorrespondenceAnalysis(Ordination):
    '''An implementation of correspondence analysis. This analysis produces embeddings for both the rows and columns
    of the matrix of counts. This analysis relies on the assumption that species exhibit unimodal response curves.
    
    Sources:
    (1) https://www.mathematica-journal.com/2010/09/20/an-introduction-to-correspondence-analysis/.
    (2) https://sw1.github.io/teaching/microbiome.html#ca 
    
    '''

    def __init__(self, scaling:int=1):
        '''Initialize a CorrespondenceAnalysis object.
        
        :param scaling: The scaling to apply to the row and column scores. With scaling 1, the distances among rows in reduced
            space approximate to their Chi-squared distance, and the rows are at the centroids of the column embeddings. With scaling 2, 
            the distances among columns in reduced space approximate to their Chi-squared distance, and the columns are at the
            centroids of the row embeddings.
        '''

        super().__init__(n_components=2, axes_labels=('CA_1', 'CA_2'))

        assert scaling in [1, 2], f'CorrespondenceAnalysis.__init__: Scaling {scaling} is invalid.'
        self.scaling = scaling
        self.inertia = None

    # TODO: I should really try to vectorize this once I confirm it is working. 
    def get_Qbar(self, P:np.ndarray, pi:np.ndarray, pj:np.ndarray, compute_inertia:bool=True)-> np.ndarray:
        '''Compute matrix Qbar, which is used both in CA and CCA. This is a centered matrix, and is similar to the Chi-squared
        distance matrix (with the exclusion of a scalar factor np.sqrt(np.sum(M))). This function optionally computes the total
        inertia.
        
        :param P: A matrix containing the relative frequencies, derived from the matrix of raw abundance data M by M / np.sum(M).
        :param pi: A vector containing the row totals of the matrix P.
        :param pj: A vector containing the column totals of the matrix P.
        :param compute_inertia: Whether or not to compute the inertia of the dataset.
        :param: A matrix Qbar, with the same shape as P. 
        '''
        E = np.outer(pi, pj) # Get the expected frequencies of each element. Outer product array is len(pi) by len(pj), and each element is pi[i] * pj[j]
        Qbar = (P - E) / np.sqrt(E)
        # for i in range(P.shape[0]):
        #     for j in range(P.shape[1]):
        #         Qbar[i, j] = P[i, j] - pi[i] * pj[j]
        #         Qbar[i, j] = Qbar[i, j] / np.sqrt( pi[i] * pj[j])

        # If specified, compute the inertia, which is the sum of squares of all values in the Qbar matrix.
        inertia = None
        if compute_inertia:
            inertia = np.sum(Qbar ** 2)
        return Qbar, inertia

    def _fit(self, M:CountMatrix=None) -> Dict[str, np.ndarray]:
        '''Carry out CorrespondenceAnalysis, following the steps presented in Numerical Ecology (Legendre and Legendre), pages 465-468.
        
        :param M: The CountMatrix object containing abundance data. 
        '''
        P = M.matrix / np.sum(M.matrix) # Compute the matrix of relative frequencies. 
        pi, pj = np.sum(Y, axis=1), np.sum(Y, axis=0) # Compute the vectors containing sums of rows and columns. 

        Qbar, self.inertia = self.get_Qbar(P, pi, pj, compute_inertia=True)

        U, svals, Uhat = np.linalg.svd(Q, compute_uv=True)
        W = np.diag(svals) # Store the singular values in a diagonal matrix.
        # Note that the diagonal matrix W.T @ W, which contains the squared singular values, contains the eigenvalues of 
        # the c x c square matrix Qbar.T @ Q (as a result of the orthonormality of U). Similarly, W @ W.T contains the eigenvalues
        # of the r x r square matrix of Qbar @ Qbar.T. So, the matrices U and Uhat contain the loadings of the columns and rows, respectively. 
        # Identical results can be obtained using eigenvalue decomposition of Qbar.T @ Qbar and Qbar @ Qbar.T.

        # Weight the singular vector matrices by the row and column weights.
        V = np.diag(1/np.sqrt(pj))  @ U
        Vhat = np.diag(1/np.sqrt(pi))  @ Uhat
        F = np.diag(1/po) @ P @ V
        Fhat = np.diag(1/pj) @ P.T @ Vhat

        if scaling == 1: # Uses matrix F for site (row) scores and V for species (column) scores. 
            row_scores = F
            col_scores = V
        if scaling == 2: # Uses matrix Fhat for species (column) scores and Vhat for site (row) scores.
            row_scores = Vhat
            col_scores = Fhat

        return {'row_scores':row_scores, 'col_scores':col_scores, 'evals':svals ** 2, 'M':M}

    def _post(self, row_scores:np.ndarray=None, col_scores:np.ndarray=None, evals:np.ndarray=None):
        
        # Eigenvalues can be used as a measure of dependence between sites and taxa for a particular ordination
        # axis. A higher value 1 - eval indicates more interdependence, I think. 
        if evals is not None:
            self.dependence = 1 - evals
        return row_scores, col_scores


class CanonicalCorrespondenceAnalysis(CorrespondenceAnalysis):
    '''Canonical, or constrained, correspondence analysis. 
    
    Sources:
    (1) https://uw.pressbooks.pub/appliedmultivariatestatistics/chapter/ca-dca-and-cca/
    '''

    # https://sw1.github.io/teaching/microbiome.html#ca

    def __init__(self, n_components:int=2, scaling:int=1, fields:List[str]=None):
        '''Initialize a CanonicalCorrespondenceAnalysis object.'''
        Ordination.__init__(self, axes_labels=('CCA_1', 'CCA_2'))

        self.scaling = scaling
        self.fields = fields

    def _fit(self, M:CountMatrix=None):
        '''Follows the algorithm in Numerical Ecology.'''
        Y = M.matrix # Extract the matrix of absolute frequencies from the CountMatrix object. 
        n, p = Y.shape # Matrix Y should have dimensions n by p, for p species collected at n sites.
        P = Y / np.sum(Y) # Compute the matrix of relative frequencies. 
        pi, pj = np.sum(P, axis=1), np.sum(P, axis=0) # Compute the vectors containing sums of rows and columns. 

        # Make sure to convert all metadata values to numeric datatype. 
        X = M.metadata[self.fields].apply(pd.to_numeric).values # Get the matrix X, which contains the environmental data. 

        def get_Xstand(X:np.ndarray):
            '''Compute the matrix Xstand, which will contain a number of rows equal to np.sum(Y). Each row
            in X is duplicated as many times as needed to make every individual organism a copy of the
            explanatory data. The matrix is then standardized.'''
            Xinfl = np.zeros((int(np.sum(Y)), len(self.fields))) 
            row_totals = np.sum(Y, axis=1).astype(int) # Get an array with the number of observations in each sample
            curr = 0
            for i in range(len(row_totals)): # Iterate over samples. 
                x = X[i, :]
                Xinfl[curr:int(curr + row_totals[i]), :] = x
                curr += int(curr + row_totals[i] )
            # Standardize X using the mean and standard deviation in each column. of Xinfl. Uses the maximum-likelihood estimator.
            # TODO: Why use maximum likelihood estimator?
            Xstand = (X - np.mean(Xinfl, axis=0, keepdims=True)) / np.std(Xinfl, ddof=0, axis=0, keepdims=True)
            return Xstand # Return the standardized Xinfl.

        Xstand = get_Xstand(X)
        Qbar, self.inertia = self.get_Qbar(P, pi, pj, compute_inertia=True)
        # Carry out weighted multiple regression using the row weights np.sqrt(D_pi)
        Yhat = np.diag(np.sqrt(pi)) @ Xstand @ np.linalg.inv(Xstand.T @ np.diag(pi) @ Xstand) @ Xstand.T @ np.sqrt(np.diag(pi)) @ Qbar
        # Compute the matrix of sums of squares and cross products. 
        S = Yhat.T @ Yhat
        # Eigen-decomposition of S to get U and Lam.
        evals, U = np.linalg.eig(S) # should I be concerned about non-real eigenvalues?
        real_and_nonzero_idxs = (np.real(evals) == evals) & (np.real(evals) > 0)
        evals, U = np.real(evals[real_and_nonzero_idxs]), np.real(U[:, real_and_nonzero_idxs])
        sort_idxs = np.argsort(evals)[::-1]
        evals, U = evals[sort_idxs], U[:, sort_idxs]
        # Define matrix Uhat, which contains the loadings of the rows of Qbar on the ordination axis.
        Uhat = (Qbar @ U) @ np.diag(1/np.sqrt(evals))

        if self.scaling == 1:
            col_scores = np.diag(1/np.sqrt(pj)) @ U
            row_scores = np.diag(1/np.sqrt(pi)) @ (Yhat @ U)
        elif self.scaling == 2:
            row_scores = np.diag(1/np.sqrt(pi)) @ (Yhat @ U) @ np.diag(1/np.sqrt(evals))
            col_scores =  np.diag(1/np.sqrt(pj)) @ U @ np.diag(1/np.sqrt(evals)) # This is the Fhat matrix defined on page 665.

        return {'row_scores':row_scores, 'col_scores':col_scores}


# TODO: Look into any recommended preprocessing when using UMAP for ecological data. 

class UMAP(Ordination):
    '''A class for using UMAP to reduce the dimensions of a CountMatrix object.'''
    # Map to convert any names to something recognized by the umap.UMAP class. 
    valid_metrics = {'bray-curtis':'braycurtis', 'euclidean':'euclidean'}

    def __init__(self, n_components:int=2, metric:str='bray-curtis'):
        # Should I transform the data for UMAP as well?
        super().__init__(n_components, axes_labels=('UMAP_1', 'UMAP_2'))

        assert metric in UMAP.valid_metrics
        
    def fit(self, M:CountMatrix) -> Dict[str, np.ndarray]:
        '''Fit the CountMatrix to the model.'''
        model = umap.UMAP(n_components=n_components, metric=UMAP.valid_metrics[metric])
        return {'row_scores':model.fit_transform(M.to_numpy()), 'col_scores':None}


class NonmetricMultidimensionalScaling(Ordination):
    '''A class for handling the non-metric multidimensional scaling of a CountMatrix object. The pre- and post-processing 
    options provided by this class, as well as the default parameters, are modeled on the metaMDS function in R's vegan package. The
    implementation of the NMDS algorithm was adapted from https://github.com/Auerilas/ecopy/blob/master/ecopy/ordination/mds.py'''
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
        super().__init__(n_components, axes_labels=('NMDS_1', 'NMDS_2'))

        assert metric in DistanceMatrix.valid_metrics, 'NonmetricMultidimensionalScaling: Dissimilarity metric is invalid.'
        self.metric = metric
        self.transform = transform
        self.max_tries, self.max_iters = max_tries, max_iters
        self.epsilon = 0.05
        self.residual_threshold = 0.01
        self.rmse_threshold = 0.005
        self.stress = None # Will store the stress of the final NMDS fit. 

        # Setting used during post-processing on NMDS solution. 
        self.half_change_scaling = half_change_scaling
        self.rotate_to_principal_components = rotate_to_principal_components

    def _prep(self, M:CountMatrix) -> Dict[str, object]:
        '''This function mimics the matrix preprocessing performed in the metaMDSdist function. If self.transform is set to True, and any 
        value in the input CountMatrix exceeds 50, then a square root transformation is applied. Additionally, if self.transform is set to true and
        any value in the input CountMatrix exceeds 9, then a Wisconsin transformation is also applied. 
        
        Following these transformations, a distance matrix is computed using the abundance data, using the metric specified in the self.metric attribute. 
        
        :param M: A CountMatrix containing raw (non-normalized) abundance data.
        :return: A dictionary containing a distance matrix/
        '''
        # In the metaMDSdist function, both transformations can be applied (this is NOT supposed to be elif). 
        if (max(M) > 50) and self.transform:
            print('Square root transformation applied.')
            M = SquareRootTransformation()(M, inplace=False)
        if (max(M) > 9) and self.transform:
            print('Wisconsin double standardization applied.')
            M = WisconsinTransformation()(M, inplace=False)

        D = DistanceMatrix(metric=self.metric).from_count_matrix(M).to_numpy()

        assert np.all(D.matrix >= 0), 'NonmetricMultidimensionalScaling._pre_nmds: All values in the DistanceMatrix must be non-negative.'
        return {'D':D}

    def _fit(self, D:np.ndarray, verbose:bool=True) -> Dict[str, np.ndarray]:
        '''Fit the data stored in a CountMatrix to the NMDS model.'''
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

        # Store the NMDS results stress.
        self.stress = stress_best

        return {'sol':sol_best}

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

        return {'stress':stress, 'sol':sol, 'n_iters':n_iters}

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
    
    def _post(self, sol:np.ndarray=None) -> Tuple[str, object]:
        '''Postprocessing procedure for NMDS. It scales and normalizes the ordination scores, and then projects
        the scores to principal components. This procedure omits some of the options provided by the vegan package in metaMDS, such
        as half-change scaling, as this did not seem to do all that much.
        
        :param sol: The NMDS solution generated in the self._fit function.
        :return: A tuple containing the row scores, which is just a modified version of the sol array. The second element
            of the tuple is None, and is just there for consistency. 
        '''
        # Normalizes the row scores such so that the centroid is moved to the zero origin and the sum of squares
        # is adjusted such that the root-mean-squared distance between points and the origin is 1. This is done in postMDS in vegan.
        centers = np.mean(sol, axis=0)
        sol = sol - centers
        # Calculate the RMS distance of centered points from the origin. 
        scaling_factor = len(sol) / np.sum(np.square(sol))
        scaling_factor = np.sqrt(scaling_factor)
        sol = sol * scaling_factor
        # Rotate the solution to principal components for easier interpretation. 
        pca = PCA(n_components=None)
        row_scores = pca.fit_transform(sol)

        return row_scores, None
    