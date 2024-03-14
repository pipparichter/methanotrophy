import unittest
import sys
sys.path.append('/home/prichter/Documents/methanotrophy/tests/')
from ordinate import *
from matrix import * 
import os
import argparse

DATA_DIR = '/home/prichter/Documents/trophy/data'

dune = rdata.read_rda(f'{DATA_DIR}/dune.rda')['dune']
dune_metadata = rdata.read_rda(f'{DATA_DIR}/dune.env.rda')['dune.env'].rename(columns={c:c.lower() for c in ['A1', 'Moisture', 'Management', 'Use', 'Manure']})


class TestNonmetricMultidimensionalScaling(unittest.TestCase):

    def setUp(self):
        '''Load in a CountMatrix and fit to different versions of the NMDS model.'''
        self.M = CountMatrix(level='species').from_pandas(dune)
        self.M.metadata = dune_metadata 
    
    def test_nmds_dimensions_are_correct(self):
        ''''''
        # Get the number of rows and columns of the row embeddings. 
        n_rows, n_cols = self.model_bray_curtis.row_scores.shape
        self.assertTrue(n_cols == self.n_components)
        self.assertTrue(n_rows == len(self.M))


    def test_nmds_procrustes(self):
        pass



class TestCorresponsenceAnalysis(unittest.TestCase):

    DATA_DIR = ''

    def setUp(self):
        '''Load in a CountMatrix and fit to different versions of the NMDS model.'''
        self.M = CountMatrix(level='asv') # For now, just run tests on the ASV matrix. 
        self.M.read_csv(os.path.join(self.DATA_DIR, 'data.csv'))
        self.M.load_metadata('metadata.csv')

        self.n_components = 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-d', help='Directory where the testing data is stored', default='/home/prichter/Documents/methanotrophy/tests', type=str)

    args = parser.parse_args()
    TestCorresponsenceAnalysis.DATA_DIR = args.data_dir
    TestNonmetricMultidimensionalScaling.DATA_DIR = args.data_dir

    unittest.main()