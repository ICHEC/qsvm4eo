import collections as col
import random as rd

import numpy as np 


def normalise_array(x, n=1, axis=0):
    """
    Normalises array between 0 and n along a given dimension.

    Parameters
    ----------
    x : np.ndarray
        An array of floats.
    n : int
        The maximum of the normalised array.
    axis : int
        The axis along which do the normalisation.

    Returns
    -------
    np.ndarray
        The normalised array
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    x_min = np.min(x, axis=axis, keepdims=True)
    x_normalised = ((x - x_min) / (x_max - x_min)) * n
    return x_normalised


def majority_vote(x, seed=24041916):
    """
    Finds the most common value of an array | list. If there is a tie,
    the value will be decided randomly.

    Parameters
    ----------
    x : np.ndarray | list
        An array of integers.
    seed : int
        Random seed used to randomly break ties.

    Returns
    -------
    int
        The most common value of the list
    """
    rd.seed(seed)
    counts = col.Counter(x)
    max_count = max(counts.values())
    ties = [k for k, v in counts.items() if v == max_count]
    most_common_value = rd.choice(ties)
    return int(most_common_value)