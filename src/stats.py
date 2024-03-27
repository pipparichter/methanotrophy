import pandas as pd 
import numpy as np
from matrix import CountMatrix
from typing import List, Union
from sklearn.linear_model import LogisticRegression, LinearRegression

# Six methods for statistically identifying and quantifying meaningful species–habitat associations are discussed. These are (1) comparison among group means (e.g. ANOVA), (2) multiple linear regression, (3) multiple logistic regression, (4) classification and regression trees, (5) multivariate techniques (principal components analysis and discriminant function analysis), and (6) occupancy modeling. Each method is described in statistical detail and associated terminology is explained. The example of habitat associations of a hypothetical beetle species (from Chapter 8) is used to further explain some of the methods. Assumption, strengths, and weaknesses of each method are discussed. Related statistical constructs and procedures such as the variance–covariance matrix, negative binomial distribution, generalized linear modeling, maximum likelihood estimation, and Bayes’ theorem are also explained. Some historical context is provided for some of the methods.

# class MultipleLogisticRegression():
#     '''An implementation of logistic regression of abundance data against one or more environmental variables. This approach assumes
#     that observations are independent. It also requires the response variable to be categorical, and if continuous, it will be binned.
    
#     Sources:
#     (1) https://stats.libretexts.org/Bookshelves/Applied_Statistics/Biological_Statistics_(McDonald)/05%3A_Tests_for_Multiple_Measurement_Variables/5.07%3A_Multiple_Logistic_Regression 
#     '''

#     def __init__(self):
#         pass

#     def fit(self, M:CountMatrix, response_vars:List[str]=None):

#         for var in response_vars:

#             y = M.get_metadata(y)

#             if response_vars.dtype in [np.float64, float]:
#                 pass

#     def _fit(self, X:np.ndarray, y:pd.Series):
#         pass


class MultipleLinearRegression():  
    '''An implementation of linear regression of abundance data against one or more environmental variables. This approach assumes a linear
    relationship between species abundance and the environmental variable.'''

    def __init__(self):
        pass

    def fit(self, M:CountMatrix, response_vars:List[str]=None):
        pass


class MutlivariateAnalysisOfVariance():
    pass 


class PermutationalMultivariateAnalysisOfVariance():
    '''
    Sources:
    (1) https://uw.pressbooks.pub/appliedmultivariatestatistics/chapter/permanova/ 
    '''
    pass


class WilcoxonDifferentialAbundance():
    pass


