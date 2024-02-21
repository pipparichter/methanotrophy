import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import seaborn as sns 
from matrix import CountMatrix
from ordinate import *
from tqdm import tqdm
from typing import NoReturn, List


def plot_rarefaction_curves(matrix:CountMatrix, n_reps:int=10, path:str=None):
    '''Plot the rarefaction curve for each sample stored in an AsvMatrix.'''

    fig, ax = plt.subplots()

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
    ax.axvline(x=smallest_sample_size, ymin=ymin, ymax=ymax, ls='--', c='gray')

    if path is not None:
        fig.savefig(path, format='PNG')

    plt.show()


def plot_correspondence_analysis(ca:CorrespondenceAnalysis, col_scale:float=0.1, title:str=None) -> NoReturn:
    '''Information about interpreting correspondence analysis plots is can be found
    here: https://www.displayr.com/interpret-correspondence-analysis-plots-probably-isnt-way-think/'''
    fig, ax = plt.subplots(1, figsize=(9, 7))
    # Grab the first two dimensions of the row and column embeddings. 
    row_embeddings, col_embeddings = ca.row_embeddings[:, :2], col_scale * ca.col_embeddings[:, :2]

    ax.scatter(row_embeddings[:, 0], row_embeddings[:, 1])
    ax.scatter(col_embeddings[:, 0], col_embeddings[:, 1])
    # Want to plot lines connecting the origin to each row point. 
    for (x, y), label in zip(row_embeddings, ca.row_labels):
        ax.plot([0, x], [0, y], c='gray', lw=1)
        # Also annotate the point with the label. 
        ax.annotate(label, (x, y))
    for (x, y), label in zip(col_embeddings, ca.col_labels):
        ax.plot([0, x], [0, y], c='lightgray', lw=1)
        # Also annotate the point with the label. 
        ax.annotate(label, (x, y), fontsize=5)    

    ax.axis('off')


def plot_nonmetric_multidimensional_scaling(nmds:NonmetricMultidimensionalScaling, title:str=None, labels:pd.Series=None, legend:bool=False) -> NoReturn:
    '''Plot the result of NMDS ordination.

    :param nmds: A NonmetricMultiDimensionalScaling object which has been fitted to a CountMatrix. 
    :param labels: A pandas Series containing labels for each scatter point. 
    :param title: A title for the plot. 
    '''

    fig, ax = plt.subplots()

    data = pd.DataFrame({'NMDS 1':nmds.embeddings[:, 0], 'NMDS 2':nmds.embeddings[:, 1]})
    data[labels.name] = labels
    
    sns.scatterplot(data=data, ax=ax, x='NMDS 1', y='NMDS 2', hue=labels.name)
    ax.set_title('' if title is None else title)

    if not legend:
        ax.legend([])



# def plot_library_size
        
