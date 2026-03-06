import numpy as np
import pandas as pd
import qsvm4eo


class ChainEmbedding:
    """
    Encodes Sentinel-2 datapoints as 1D atom chains for neutral atom quantum computing.
    Each feature maps to one atom placed at (c * normalized_value, 0).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset used. 
    c : float
        Scaling constant (µm). Controls inter-atom spacing.
        Must be tuned so that no two atoms are closer than min_distance (5 µm).
        Must be in the range [6, 24].
    """

    def __init__(
        self,
        df: pd.DataFrame
    ):
        self.B0x_normalised = qsvm4eo.utils.normalise_array(df[["B02", "B03", "B04", "B08"]].to_numpy())

    def embed(self, c=10) -> list[np.ndarray]:
        x = -36 + c * np.arange(4)
        return np.stack((np.broadcast_to(x, self.B0x_normalised.shape), self.B0x_normalised), axis=2)