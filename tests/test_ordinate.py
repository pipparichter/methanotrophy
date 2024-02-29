import unittest
import sys
sys.path.append('../src')
from ordinate import *
from matrix import * 



class TestNonmetricMultidimensionalScaling(unittest.TestCase):

    def setUp(self):
        '''Load in a CountMatrix and fit to different versions of the NMDS model.'''
        self.M = CountMatrix(level='asv') # For now, just run tests on the ASV matrix. 
        self.M.read_csv('data.csv')
        self.M.load_metadata('metadata.csv')

        self.n_components = 2
        self.model_bray_curtis = NonmetricMultidimensionalScaling(n_components=self.n_components, metric='bray-curtis')
        self.model_chi_squared = NonmetricMultidimensionalScaling(n_components=self.n_components, metric='chi-squared')

        self.model_bray_curtis.fit(self.M)
        self.model_chi_squared.fit(self.M)
    
    def test_nmds_bray_curtis_dimensions_are_correct(self):
        ''''''
        # Get the number of rows and columns of the row embeddings. 
        n_rows, n_cols = self.model_bray_curtis.row_scores.shape
        self.assertTrue(n_cols == self.n_components)
        self.assertTrue(n_rows == len(self.M))

    def test_nmds_chi_squared_dimensions_are_correct(self):
        ''''''
        # Get the number of rows and columns of the row embeddings. 
        n_rows, n_cols = self.model_chi_squared.row_scores.shape
        self.assertTrue(n_cols == self.n_components)
        self.assertTrue(n_rows == len(self.M))

    def test_nmds_fit_surface(self):

        y = self.M.get_metadata('uninterested') # This is the continuous variable. 
        self.model_bray_curtis.fit_surface(y)






if __name__ == '__main__':
    unittest.main()