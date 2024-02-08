'''Functions for reading and writing methanotrophy data.'''
import pandas as pd 
import numpy as np
import re
import os
from tqdm import tqdm
from typing import NoReturn


def dataframe_from_metadata(path:str) -> pd.DataFrame:
    '''Load the sample metadata.'''
    metadata = pd.read_csv(path)
    metadata.columns = [col.lower() for col in metadata.columns]
    metadata = metadata[~metadata.serial_code.isin(['PCR blank', 'Extr blank', 'KML'])] # Drop the outliers. 
    metadata.serial_code = metadata.serial_code.apply(int) # Convert serial code to integers.
    metadata.soil_depth = metadata.soil_depth.str.lower()
    
    return metadata


def dataframe_from_taxonomy(path:str) -> pd.DataFrame:
    '''Load the ASV taxonomy information.'''
    taxonomy = pd.read_csv(path, delimiter='\t', index_col=0)
    # Standardize the unclassified taxa (they are entered as TAXA_NAME_unclassified)
    taxonomy = taxonomy.map(lambda s : 'unclassified' if 'unclassified' in s else s)
    # Not sure if this is safe to do. 
    # taxonomy = taxonomy.applymap(lambda s : re.sub('_[0-9]+', '', s))

    for col in taxonomy.columns:
        # taxonomy[col + '_sub'] = taxonomy[col].apply(lambda s : int(s.split('_')[-1]) if '_' in s else 0)
        p = '([a-zA-Z0-9_]+)_([0-9]+)' # Pattern to match, all taxa ending in _{number}{number}
        taxonomy[col + '_sub'] = taxonomy[col].apply(lambda s : int(re.match(p, s).group(2)) if not (re.match(p, s) is None) else 0)
        taxonomy[col] = taxonomy[col].apply(lambda s : re.match(p, s).group(1) if not (re.match(p, s) is None) else s)
    
    taxonomy['asv'] = taxonomy.index
    taxonomy.index = np.arange(len(taxonomy))
    return taxonomy


def dataframe_from_counts(path:str) -> pd.DataFrame:
    '''Load the ASV count information.'''
    counts = pd.read_csv(path, delimiter='\t', index_col=0)
    num_entries = counts.shape[0] * counts.shape[1]

    counts['asv'] = counts.index
    counts.index = np.arange(len(counts))
    # Convert columns to a categorical 'sample' variable. 
    counts = counts.melt(id_vars=['asv'], value_name='count', var_name='sample')
    assert len(counts) == num_entries, 'io.load_asv_data: Some data was lost!' # Make sure no data was lost when reformatting the DataFrame. 
    # Throw out the weird samples.
    counts = counts[~counts['sample'].isin(['HDK-DNAexNegLot169030916-30cyc', 'HDK-MAR-PCR-BLANK'])]

    return counts
