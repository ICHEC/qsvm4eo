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
    """

    def __init__(
        self,
        df: pd.DataFrame
    ):
        self.B0x_normalised = qsvm4eo.utils.normalise_array(df[["B02", "B03", "B04", "B08"]].to_numpy())

    def embed(self, c1=36, c2=10) -> list[np.ndarray]:
        """
        Parameters
        ----------
        c1,c2 : float 
            Constants, need to be chosen so that the device constraints are verified. 
            This means that |-c1|≤ 38 and |-c1 + 3c2| ≤ 38
        """
        x = -c1 + c2 * np.arange(4)
        return np.stack((np.broadcast_to(x, self.B0x_normalised.shape), self.B0x_normalised), axis=2)