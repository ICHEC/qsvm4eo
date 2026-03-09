import numpy as np


class RadialEmbedding:
    """
    Embedder for transforming the feature vectors into qubit coordinates
    using the Radial Embedding.

    Parameters
    ----------
    df: pandas.Datafrea
    Notes
    -----
    If the feature vector is `(x1, x2, ..., xn)` (where n is the number of features),
    then we embed each feature `xi` as a qubit with coordinates

        `xi=[ ri cos(2π i/n) , ri sin (2π i/n)]`

    where ri is the radius given by ri=(xi+a)b.
    Here a and b are hyperparameters (the shift and scaling respectively)
    which are chosen to prevent the qubits being too close to each other or too far away.
    The angle between two adjacent points is 2π /n.

    The radii are scaled so that they are all between `shift*scaling` and `(1+shift)*scaling`.
    E.g. choosing `shift=1` and `scaling=5` implies all the radii are between 5 and 10.
    """

    def __init__(self, df):
        """
        Parameters
        ----------
        df : pd.DataFrame
        A dataset containing (at least) the following columns
        ['Label', 'B02', 'B03', 'B04', 'B08']
        """
        self.B0x = df[["B02", "B03", "B04", "B08"]].to_numpy()
        self.n_features = self.B0x.shape[1]
        angles = np.linspace(0, 2 * np.pi, self.n_features, endpoint=False)
        self.unit_circle = np.array([np.cos(angles), np.sin(angles)]).T

    def embed(self, shift=1.0, scaling=5.4):
        """
        Generate the embedding

        Parameters
        ----------
        shift : float
            The shift hyperparameter.
        scaling : float
            The scaling hyperparameter.
        """
        max_feature = np.max(self.B0x)
        radius = (self.B0x / max_feature + shift) * scaling
        coords = radius[:, :, None] * self.unit_circle[None, :, :]
        return coords
