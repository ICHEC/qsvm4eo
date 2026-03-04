import numpy as np
import qsvm4eo


class GeneticEmbedding:
    """
    Class for implementing the GeneticEmbedding.

    In this embedding, each graph is generated using a genetic algorithm.
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
            Index of the data sample being embedded.
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

    def embed(
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
        Run the genetic algorithm embedding for each data sample.

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
