from sklearn.svm import SVC


class QSVM:
    """
    A class for the Quantum Support Vector Machine (QSVM).

    Parameters
    ----------
    regularization : float
        The regularization parameter.
        Defaults to 1.
    """

    def __init__(self, regularization=1.0):
        self.model = SVC(kernel="precomputed", C=regularization)

    def train(self, gram_train, y_train):
        """
        Train the model and return the training accuracy.
        """
        self.model.fit(gram_train, y_train)
        return self.model.score(gram_train, y_train)

    def predict(self, gram_test):
        """Returns predicted labels for a given gram matrix."""
        return self.model.predict(gram_test)
