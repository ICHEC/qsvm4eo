
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



def load_data(data_path, num_features, scale_features=False):
    """
    Load the data.

    Parameters
    ----------
    data_path : str
        The path to the data directory.
    num_features : int
        The number of training features. Either 4 or 8.
    scale_features : bool
        Whether to scale the features such that the mean
        is 0 and variance is 1. Defaults to False.
    """
    if num_features == 4:
        features = ["B02", "B03", "B04", "B08"]
    elif num_features == 8:
        features = ["B02", "B03", "B04", "B08", "NDVI", "EVI", "SAVI", "NDWI"]
    else:
        raise Exception("`num_features` must be 4 or 8")

    data_train = pd.read_csv(f"{data_path}/data/train_32.csv", header=0)
    x_train = np.array(data_train[features].values, dtype=float)
    y_train = data_train["Label"].values

    data_test = pd.read_csv(f"{data_path}/data/test_32.csv", header=0)
    x_test = np.array(data_test[features].values, dtype=float)
    y_test = data_test["Label"].values

    if scale_features:
        # We remove mean and scale to unit variance
        scaler = StandardScaler()

        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test


