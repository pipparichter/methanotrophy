import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
from matrix import *
import pickle
import argparse





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', help='The path to the pickled CountMatrix object.', type=str)
    parser.add_argument('out_path', help='The path specifying where to write the pickled NMDS object.', type=str)
    parser.add_argument('-f', '--figure-path', help='The path specifying where to write the NMDS plot.', type=str)

    args = parser.parse_args()

    norm = ConstantSumScaling()
    
    norm(tm)

    nmds = NonmetricMultidimensionalScaling(n_components=2, metric='bray-curtis')
    nmds.fit(tm)

    plot_nonmetric_multidimensional_scaling(nmds, labels=tm.get_metadata('flux_ch4'))
    plt.savefig('/home/prichter/Documents/methanotrophy/figures/nmds_flux_ch4.png')