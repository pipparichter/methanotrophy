import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
from matrix import *
import pickle
import argparse


# Generating the count matrix is currently extremely slow. Possibly a way to speed it up?
# m = create_asv_matrix(pd.read_csv(f'{DATA_DIR}/data.csv'))
# m.filter_read_depth(5000)

tm = m.get_taxonomy_matrix()

norm = ConstantSumScaling()
norm(tm)

nmds = NonmetricMultidimensionalScaling(n_components=2, metric='bray-curtis')
nmds.fit(tm)

plot_nonmetric_multidimensional_scaling(nmds, labels=tm.get_metadata('flux_ch4'))
plt.savefig('/home/prichter/Documents/methanotrophy/figures/nmds_flux_ch4.png')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()