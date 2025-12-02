import numpy as np


class RadialEncoding:
    """
    Encoder for transforming the feature vectors into qubit coordinates
    using the Radial Encoding.

    Parameters
    ----------
    max_feature : float
        The largest feature in the training set.
    shift : float
        The shift hyperparameter.
    scaling : float
        The scaling hyperparameter.
    n_features : int
        The number of features.

    Notes
    -----
    If the feature vector is `(x1, x2, ..., xn)` (where n is the number of features),
    then we encode each feature `xi` as a qubit with coordinates

        `xi=[ ri cos(2π i/n) , ri sin (2π i/n)]`

    where ri is the radius given by ri=(xi+a)b.
    Here a and b are hyperparameters (the shift and scaling respectively)
    which are chosen to prevent the qubits being too close to each other or too far away.
    The angle between two adjacent points is 2π /n.

    The radii are scaled so that they are all between `shift*scaling` and `(1+shift)*scaling`.
    E.g. choosing `shift=1` and `scaling=5` implies all the radii are between 5 and 10.
    """

    def __init__(self, max_feature, shift, scaling, n_features):
        self.shift = shift * scaling
        self.scaling = scaling / max_feature

        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
        self.unit_circle = np.array([np.cos(angles), np.sin(angles)]).T

    def encode(self, x):
        """
        Enode a data point into a set of qubit coordinates.

        Parameters
        ----------
        x : np.ndarray
            The feature vector to be encoded.

        Returns
        -------
        np.ndarray
            The qubit coordinates.
        """
        radius = x * self.scaling + self.shift
        return radius[:, None] * self.unit_circle
