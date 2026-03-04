import numpy as np
import scipy as scp
import skimage as ski

import qsvm4eo


class ConvolutionalEmbedding:
    """
    Class for implementing the ConvolutionalEmbedding.

    In this embedding, all datapoints are first placed on a grid according to their latitudes and longitudes.
    We then convolve the points in square blocks of size 2 ** n_convoluted_side. Each block becomes a graph
    that will be used as input to the analog QC.

    To assign coordinates to the nodes of each graph, we convert the RGB values associated with each node
    into HSV (Hue, Saturation, Value). The Hue component (H), interpreted as an angle, is then used to generate
    the Cartesian coordinates of each node in the graph. We apply scaling factors to the radial component of the
    polar coordinates and include angle offsets to ensure that the resulting embeddings are compatible with the QC system.
    """

    def __init__(self, df):
        """

        Parameters
        ----------
        df : pd.DataFrame
            A dataset containing (at least) the following columns ['Label', 'Latitude', 'Longitude', 'B02', 'B03', 'B04']:
        """
        self.df = df
        self.colour_transformation()
        self.rank_sort_coordinates()

    def colour_transformation(self):
        """
        Transforms our RGB coordinates to HSV coordinates
        """
        rgb_coordinates = qsvm4eo.normalise_array(
            self.df[["B04", "B03", "B02"]].to_numpy()
        )
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
        self.df["lat_rank"] = (
            scp.stats.rankdata(self.df["Latitude"], method="dense") - 1
        )
        self.df["lon_rank"] = (
            scp.stats.rankdata(self.df["Longitude"], method="dense") - 1
        )
        self.df.sort_values(by=["lat_rank", "lon_rank"], inplace=True)

    def convolute_in_squares(self, n_convoluted_side=2):
        """
        Convolutes data points in squares.

        Parameters
        ----------
        n_convoluted_side : int
            Number of points on each side of the square. (default is 2)
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

    def hsv_embedding(self, n_convoluted_side=2, scaling=37.0):
        """
        Computes the coordinates of the nodes of each of the graphs in the dataset.
        It does so by assigning one angle to each of the nodes. This angle
        will be the HUE angle plus an offset factor so that our graphs are embeddable
        in the analog device. Then we use those angles to change from polar
        to cartesian coordinates.

        Parameters
        ----------
        n_convoluted_side : int
            Number of points on each side of the square (default is 2).
        scaling : float
            Radial coordinate to apply to our coordinates. (default is 37).

        Returns
        -------
        list[np.ndarray]
            A list containing arrays with the graphs coordinates and labels.
        """
        self.convolute_in_squares(n_convoluted_side=n_convoluted_side)
        N, M, _ = self.convoluted_coordinates.shape
        unit_circle_division = (2 * np.pi) / (n_convoluted_side**2 * 2)
        angles_normalised = np.mod(
            self.convoluted_coordinates[:, :, 0], unit_circle_division
        )
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
