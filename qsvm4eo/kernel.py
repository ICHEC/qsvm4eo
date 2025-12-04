import numpy as np
from scipy.spatial import distance
import math


class Kernel:
    """
    Class for computing gram matrices based on similarity of
    the probability distributions.

    Parameters
    ----------
    mu : float
        Hyperparameter for the gram matrices.
    """

    def __init__(self, mu=0.5):
        self.mu = mu

    def compute_gram_train(self, p_train):
        """
        Compute the training gram matrix.

        Parameters
        ----------
        p_train : np.ndarray
            The probability distributions for the training data.

        Returns
        -------
        np.ndarray
            The training gram matrix.
        """
        n_train = len(p_train)
        gram = np.ones((n_train, n_train))

        for i in range(n_train - 1):
            for j in range(i + 1, n_train):
                gram[i, j] = gram[j, i] = _distance(p_train[i], p_train[j], self.mu)
        return gram

    def compute_gram_test(self, p_test, p_train):
        """
        Compute a test gram matrix.

        Parameters
        ----------
        p_test : np.ndarray
            The probability distributions for the test data.
        p_train : np.ndarray
            The probability distributions for the training data.

        Returns
        -------
        np.ndarray
            The test gram matrix.
        """
        n_test = len(p_test)
        n_train = len(p_train)
        gram = np.ones((n_test, n_train))

        for i in range(n_test):
            for j in range(n_train):
                gram[i, j] = _distance(p_train[i], p_test[j], self.mu)
        return gram


def _distance(p1, p2, mu):
    return np.exp(-mu * np.abs(p1 - p2).sum())


def compute_excitation_count(probs):
    """Compute the number of excitations from the state probabilities."""
    num_features = int(np.log2(probs.shape[1]))
    excitations = np.zeros((probs.shape[0], num_features + 1))

    for i in range(2**num_features):
        # the number of excitations
        n_excit = np.binary_repr(i, num_features).count("1")
        excitations[:, n_excit] += probs[:, i]
    return excitations
