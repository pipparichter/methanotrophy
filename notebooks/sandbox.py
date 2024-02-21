import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
from utils import * 
from matrix import *
from norm import * 
from ordinate import *
from plot import *
import os

DATA_DIR = '/home/prichter/Documents/data/methanotrophy'

def create_asv_matrix(df:pd.DataFrame) -> pd.DataFrame:
    '''Convert the DataFrame containing the sample metadata and ASV counts into an ASV table (with columns
    as ASVs, and rows as samples). Each cell contains the raw count for that particular ASV in the sample.''' 
    # Accumulate all of the metadata into a separate DataFrame. 
    metadata = df[[col for col in df.columns if col not in ['count']]]

    df = df[['serial_code', 'asv', 'count']]
    df = df.groupby(by=['serial_code', 'asv']).sum()
    df = df.reset_index() # Converts the multi-level index to categorical columns. 
    df = df.pivot(columns=['asv'], index=['serial_code'], values=['count'])
    # Reset column labels, which were weird because of the multi-indexing. 
    df.columns = df.columns.get_level_values('asv').values

    return AsvMatrix(df, metadata=metadata)


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