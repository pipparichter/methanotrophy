import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import seaborn as sns 
from matrix import AsvMatrix
from tqdm import tqdm



def plot_rarefaction_curves(matrix:AsvMatrix, n_reps:int=10, path:str=None):
    '''Plot the rarefaction curve for each sample stored in an AsvMatrix.'''

    fig, ax = plt.subplots(1)


    # Compute the rarefaction curve for each sample. 
    for i in tqdm(range(matrix.shape[0]), desc='plot.plot_rarefaction_curves'):
        # Get a range of 50 sample sizes, between 1 and the total number of observations in the sample.
        n_vals = np.linspace(1, np.sum(matrix[i]), num=50, dtype=int)
        n_species = np.zeros(len(n_vals)) # Initialize an array to store the mean species count in each subsample.
        # n_species = np.apply_along_axis(lambda n : np.count_nonzero(matrix._sample(i, n[0])), 1, n_vals)
        for _ in range(n_reps):
            n_species += np.array([matrix._sample(i, n, species_count_only=True) for n in n_vals])
        
        n_species = n_species / n_reps # Take the average of the number of species drawn for each sample size. 
        ax.plot(n_vals, n_species)

    ax.set_title('Rarefaction curves')
    ax.set_xlabel('subsample size')
    ax.set_ylabel('number of unique ASVs')

    # Plot a vertical line marking the size of the smallest sample in the dataset. 
    smallest_sample_size = min(matrix.matrix.sum(axis=1))
    ymin, ymax = ax.get_ylim()
    ax.vline(x=smallest_sample_size, ymin=ymin, ymax=ymax, ls='--', c='gray')

    if path is not None:
        fig.savefig(path, format='PNG')

    plt.show()
        
