'''Tests for the classes defined in the matrix.py file.'''
import sys
sys.path.append('/home/prichter/Documents/methanotrophy/src')
import unittest
from matrix import Matrix, CountMatrix, DistanceMatrix, _reformat_csv
import numpy as np
import argparse
import os
import rdata

DATA_DIR = '/home/prichter/Documents/trophy/data'

dune = rdata.read_rda(f'{DATA_DIR}/dune.rda')['dune']
dune_metadata = rdata.read_rda(f'{DATA_DIR}/dune.env.rda')['dune.env'].rename(columns={c:c.lower() for c in ['A1', 'Moisture', 'Management', 'Use', 'Manure']})


class TestMatrix(unittest.TestCase):
    '''Tests for core operations for working with Matrix objects defined in the Matrix class.'''

    def setUp(self):
        self.M = CountMatrix(level='species').from_pandas(dune)
        self.M.metadata = dune_metadata 


class TestCountMatrix(unittest.TestCase):

    def setUp(self):
        self.M = CountMatrix(level='species').from_pandas(dune)
        self.M.metadata = dune_metadata 

    def test_count_matrix_no_negative_entries(self):
        pass

    def test_count_matrix_no_zero_row_counts(self):
        '''An edge case test to make sure that when a CountMatrix is instantiated with count data with empty rows, 
        the constructor raises a warning and filters out the empty rows.'''

        pass

    def test_count_matrix_no_zero_col_counts(self):
        '''An edge case test to make sure that when a CountMatrix is instantiated with count data with empty columns, 
        the constructor raises a warning and filters out the empty columns.'''
        pass


class TestDistanceMatrix(unittest.TestCase):

    def setUp(self):

        self.M = CountMatrix(level='species').from_pandas(dune)
        self.M.metadata = dune_metadata 
        self.n_samples, self.n_asvs = self.M.shape()

        self.D = [] # A list of DistanceMatrix objects derived from M. 
        for metric in DistanceMatrix.valid_metrics:
            # Store the metric so it can be accessed later. 
            metrics.append(metric)
            # Instantiate distance matrices from M and its transpose.
            self.D.append(DistanceMatrix(metric=metric).from_count_matrix(self.M))
            self.D_T.append(DistanceMatrix(metric=metric).from_count_matrix(self.M_T))

    def test_distance_matrix_dimensions_are_correct(self):
        '''Check to make sure the dimensions of the DistanceMatrix objects are correct.'''
        for D in self.D:
            self.assertTrue(D.shape() == (self.n_samples, self.n_samples))

    def test_distance_matrix_is_symmetric(self):
        '''Check to make sure DistanceMatrix objects created using the Bray-Curtis dissimilarity metric
        store symmetric matrices.'''
        # If DistanceMatrix objects are symmetric, then they should be equal to the transpose.
        for D in self.D:
            self.assertTrue(D == D.transpose()) 

    def test_distance_matrix_diagonals_are_zero(self):
        '''Make sure the diagonals of the matrices stored in the DistanceMatrix object are zeros.'''
        for D in self.D:
            self.assertTrue(np.all(np.diag(D) == 0))

    def test_distance_matrix_zero_when_samples_are_identical(self):
        pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-d', help='Directory where the testing data is stored', default='/home/prichter/Documents/methanotrophy/tests', type=str)

    args = parser.parse_args()
    TestCountMatrix.DATA_DIR = args.data_dir
    TestDistanceMatrix.DATA_DIR = args.data_dir

    unittest.main()