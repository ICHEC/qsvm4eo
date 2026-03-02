import numpy as np
import qsvm4eo
import skimage as ski
import scipy as scp


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


class ConvolutionalEncoding:
    """
    Class for implementing the ConvolutionalEncoding.

    In this encoding, all datapoints are first placed on a grid according to their latitudes and longitudes.
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

    def hsv_encoding(self, n_convoluted_side=2, scaling=37.0):
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


class GeneticEncoding:
    """
    Class for implementing the GeneticEncoding.

    In this encoding, each graph is generated using a genetic algorithm.
    The genetic algorithm optimises the node positions so that each graph has
    as much similarity as possible to the original datapoints and the device
    constraints are being verified.
    """

    def __init__(self, df):
        """
        Parameters
        ----------
        df : pd.DataFrame
        A dataset containing (at least) the following columns
        ['Label', 'B02', 'B03', 'B04', 'B08']
        """
        self.max_radius = 38.0
        self.k = 4
        self.d_min = 5.0
        self.df = df
        B0x = df[["B02", "B03", "B04", "B08"]].to_numpy()
        self.B0_norm = qsvm4eo.utils.normalise_array(B0x, self.max_radius)

    def init_population(self, pop_size):
        """
        Randomly initialize a population of individuals inside the circle.

        Parameters
        ----------
        pop_size : int
            Number of individuals in the population.

        Returns
        -------
        numpy.ndarray
            Array of shape (pop_size, k, 2) representing node coordinates.
            Each individual contains `k` 2D coordinates (x, y).
        """
        r = np.random.rand(pop_size, self.k) * self.max_radius
        theta = np.random.rand(pop_size, self.k) * 2 * np.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        coords = np.stack((x, y), axis=2)
        return coords

    def fitness(self, ind, idx, alpha):
        """
        Compute the fitness value of an individual.

        The fitness function penalizes:
        1. Nodes outside the max radius allowed.
        2. Nodes closer than the minimum allowed distance.
        3. Differences between node radii and normalized band magnitudes.

        Parameters
        ----------
        ind : numpy.ndarray
            Individual of shape (k, 2) representing node coordinates.
        idx : int
            Index of the data sample being encoded.
        alpha : list or tuple of length 3
            Weighting coefficients for:
            [boundary_penalty, node_distance_penalty, band_norm_penalty].

        Returns
        -------
        float
            Negative weighted penalty (higher is better).
        """
        # 1. Node-to-centre penalty.
        norm = np.linalg.norm(ind, axis=1)
        penalty_norm = np.sum(np.maximum(0.0, norm - self.max_radius))

        # 2. Between-nodes distance penalty.
        diff = ind[:, None, :] - ind[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        i, j = np.triu_indices(self.k, k=1)
        penalty_distance_nodes = np.sum(np.maximum(0.0, self.d_min - dists[i, j]))

        # 3. bands-norm penalty
        penalty_bands_norm = np.sum(np.abs(self.B0_norm[idx] - norm))

        return -(
            alpha[0] * penalty_norm
            + alpha[1] * penalty_distance_nodes
            + alpha[2] * penalty_bands_norm
        )

    def tournament_selection(self, pop, fit, n, k_tourn=3):
        """
        Perform tournament selection.

        Parameters
        ----------
        pop : numpy.ndarray
            Current population.
        fit : numpy.ndarray
            Fitness values corresponding to the population.
        n : int
            Number of individuals to select.
        k_tourn : int, default=3
            Number of participants per tournament.

        Returns
        -------
        list
            Indices of selected individuals.
        """
        inds = []
        for _ in range(n):
            participants = np.random.choice(len(pop), k_tourn, replace=False)
            best = participants[np.argmax(fit[participants])]
            inds.append(best)
        return inds

    def sbx(self, parent1, parent2, eta=15):
        """
        Simulated Binary Crossover (SBX).

        Parameters
        ----------
        parent1 : numpy.ndarray
            First parent of shape (k, 2).
        parent2 : numpy.ndarray
            Second parent of shape (k, 2).
        eta : float, default=15
            Distribution index controlling offspring spread.
            Higher values produce children closer to parents.

        Returns
        -------
        tuple of numpy.ndarray
            Two offspring individuals of shape (k, 2).
        """
        parent1 = parent1.flatten()
        parent2 = parent2.flatten()
        child1 = np.empty_like(parent1)
        child2 = np.empty_like(parent2)
        for i in range(len(parent1)):
            if np.random.rand() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    x1, x2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])
                    rand = np.random.rand()
                    beta = (2 * rand) ** (1 / (eta + 1))
                    child1[i] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                    child2[i] = 0.5 * ((1 + beta) * x2 + (1 - beta) * x1)
                else:
                    child1[i] = parent1[i]
                    child2[i] = parent2[i]
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
        return child1.reshape(4, 2), child2.reshape(4, 2)

    def mutate(self, ind, sigma, p):
        """
        Apply Gaussian mutation to an individual.

        Parameters
        ----------
        ind : numpy.ndarray
            Individual of shape (k, 2).
        sigma : float
            Standard deviation of Gaussian noise.
        p : float
            Probability of mutating each coordinate.

        Returns
        -------
        numpy.ndarray
            Mutated individual.
        """
        mask = (
            np.random.rand(*ind.shape) < p
        )  # Decides which coordinates to randomly mutate.
        gauss = np.random.normal(0, sigma, size=ind.shape)
        ind[mask] += gauss[mask]
        return ind

    def encode(
        self,
        alpha=[10, 10, 1],
        seed=1916,
        pop_size=2000,
        generations=200,
        sigma_pos=0.5,
        p_mut=0.2,
        elitism=True,
    ):
        """
        Run the genetic algorithm encoding for each data sample.

        Parameters
        ----------
        alpha : list of length 3
            Fitness weighting coefficients.
        seed : int
            Random seed for reproducibility.
        pop_size : int
            Population size.
        generations : int
            Number of evolutionary generations.
        sigma_pos : float
            Mutation standard deviation.
        p_mut : float
            Mutation probability per coordinate.
        elitism : bool
            If True, preserve best individual each generation.

        Returns
        -------
        list
            List of best individuals (graphs) for each data sample.
            Each element has shape (k, 2).
        """
        best_graphs = []
        for idx in range(self.B0_norm.shape[0]):
            print("item index : ", idx)
            population = self.init_population(pop_size)

            for gen in range(generations):
                fit = np.array([self.fitness(ind, idx, alpha) for ind in population])
                sel_idx = self.tournament_selection(population, fit, pop_size)
                sel = population[sel_idx]

                # produce offspring
                offspring = []
                for i in range(0, pop_size, 2):
                    p1, p2 = sel[i], sel[(i + 1) % pop_size]
                    c1, c2 = self.sbx(p1, p2)
                    offspring.append(self.mutate(c1.copy(), sigma_pos, p_mut))
                    offspring.append(self.mutate(c2.copy(), sigma_pos, p_mut))
                offspring = np.array(offspring[:pop_size])

                # elitism
                if elitism:
                    best_idx = np.argmax(fit)
                    offspring[0] = population[best_idx].copy()

                population = offspring

                if gen % 30 == 0:
                    print(f"Gen {gen:3d} | best fitness {np.max(fit): .2f}")

            best_graphs.append(offspring[0])

        return best_graphs
