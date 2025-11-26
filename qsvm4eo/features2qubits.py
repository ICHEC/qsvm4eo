import numpy as np
import qsvm4eo
import skimage as ski
import scipy as scp 


class RadialEncoding:
    """
    `max_feature` is the largest feature in the training set.
    The radii are scaled so that they are all between `shift*scaling` and `(1+shift)*scaling`.
    E.g. choosing `shift=1` and `scaling=5` so that all the radii are between 5 and 10.
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
    

class ConvolutionalEncoding:
    """
    Class for implementing the ConvolutionalEncoding
    """
    def __init__(self, df):
        self.df = df 
        self.colour_transformation()
        self.rank_sort_coordinates()

    def colour_transformation(self):
        """
        Transforms our RGB coordinates to HSV coordinates
        """
        rgb_coordinates = qsvm4eo.normalise_array(self.df[["B04", "B03", "B02"]].to_numpy())
        ## Colour transformation. We add a new dimension to fit in the skimage input.
        hsv_coordinates = ski.color.rgb2hsv(rgb_coordinates[np.newaxis, :, :])[0] 
        self.df["hsv_coordinates"] = list(hsv_coordinates)

    def rank_sort_coordinates(self):
        """
        Ranks and sorts coordinates so that we have values between
        0 and N - 1 being N the number of points in our dataset.
        The dataset will be ordered with values depending on 
        latitude and longitude. 
        """
        self.df["lat_rank"] = scp.stats.rankdata(self.df["Latitude"], method="dense") - 1
        self.df["lon_rank"] = scp.stats.rankdata(self.df["Longitude"], method="dense") - 1
        self.df.sort_values( 
            by=["lat_rank", "lon_rank"], inplace=True
        )

    def convolute_in_squares(self, n_convoluted_side=2):
        """
        Convolutes data points in squares.

        Parameters
        ----------
        n_convoluted_side : int
            Number of points on each side of the square.
        """
        n_grid_side = max(
            self.df["lat_rank"] + 1
        )  # Number of points per side in the grid
        convoluted_coordinates = []
        convoluted_labels = []
        for i in range(0, n_grid_side, n_convoluted_side):
            for j in range(0, n_grid_side, n_convoluted_side):
                mask = (
                    (self.df["lat_rank"] >= i)
                    & (self.df["lat_rank"] < i + n_convoluted_side)
                    & (self.df["lon_rank"] >= j)
                    & (self.df["lon_rank"] < j + n_convoluted_side)
                )
                convoluted_coordinates.append(
                    np.stack(self.df["hsv_coordinates"][mask].to_numpy())
                )
                convoluted_labels.append(
                    qsvm4eo.majority_vote(np.stack(self.df["Label"][mask].to_numpy()))
                )
        self.convoluted_coordinates = np.stack(convoluted_coordinates)
        self.convoluted_labels = convoluted_labels

    
    def hsv_encoding(self, n_convoluted_side=2, scaling=1):
        self.convolute_in_squares(n_convoluted_side=n_convoluted_side)
        N, M, _ = self.convoluted_coordinates.shape
        unit_circle_division = (2 * np.pi) / (n_convoluted_side ** 2 * 2)
        angles_normalised = np.mod(self.convoluted_coordinates[:, :, 0], unit_circle_division)
        # Compute offsets for the angles
        offsets = 2 * np.arange(n_convoluted_side**2) * unit_circle_division
        offsets = offsets[:M]
        # Apply offsets
        angles_divided = angles_normalised + offsets
        
        # Convert to Cartesian coordinates
        x = scaling * np.cos(angles_divided)
        y = scaling * np.sin(angles_divided)
        hsv_coordinates = np.stack((x, y), axis=2)
        return hsv_coordinates, self.convoluted_labels






