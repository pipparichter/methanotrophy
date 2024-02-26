import pandas as pd 
import numpy as np
from matrix import CountMatrix


class AnalysisOfCovariance():
    pass 


class WilcoxonDifferentialAbundance():
    pass


# class AncomDirectionalMultiplePairwiseTesting():
#     '''https://academic.oup.com/biometrics/article/66/2/485/7331958'''
    

#     def get_p_values_nonparametric(delta:np.array):
#         '''Calculates the p-values without assuming anything about the underlying distribution.'''
#         # In this case, the "null" case is zero, indicating no difference in the expression of the taxon between
#         # the reference sample and every other sample. 
#         pass

#     def __call__(self, matrix:CountMatrix):

#         # Get a matrix for which each row is a taxonomical group or ASV, and each column is the abundance of 
#         # that group in a particular sample. 
#         matrix = matrix.matrix.T

#         p = matrix.shape[1] # Number of samples, or "ordered groups."
#         m = matrix.shape[0] # Number of variables, taxa in this case. 

#         for j in range(m): # Iterate over the taxa. 

#             for i in range(p): # Iterate over the samples. 

#                 mu_ij = matrix[j, i]
#                 # The delta vector contains the difference between the normalized abundance of taxon j in sample
#                 # i relative to its abundance in every other sample.
#                 delta_j = matrix[j, np.arange([x for x in range(p) if x != i])] - mu_ij



class CanonicalCorrelationAnalysis():
    pass