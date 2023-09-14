import numpy as np

class NaiveBayes:
    def __init__(self, method='gaussian'):
        self.method = method
        self.classes = None
        self.statistics = {}

    def fit(self, X, y):
        """
        Train the Naive Bayes model.

        Parameters:
        - X: Features (numpy array or similar data structure)
        - y: Target labels (numpy array or similar data structure)
        """
        self.classes = list(set(y))

        if self.method == 'gaussian':
            self._fit_gaussian(X, y)
        elif self.method == 'multinomial':
            self._fit_multinomial(X, y)
        elif self.method == 'bernoulli':
            self._fit_bernoulli(X, y)
        else:
            raise ValueError(f"Method {self.method} not recognized.")

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters:
        - X: Features to predict (numpy array or similar data structure)

        Returns:
        - Predicted labels
        """
        if self.method == 'gaussian':
            return self._predict_gaussian(X)
        elif self.method == 'multinomial':
            return self._predict_multinomial(X)
        elif self.method == 'bernoulli':
            return self._predict_bernoulli(X)
        else:
            raise ValueError(f"Method {self.method} not recognized.")

    def _fit_gaussian(self, X, y):
        """
        Fit the model for Gaussian Naive Bayes.
        """
        for c in self.classes:
            X_c = X[y == c]
            self.statistics[c] = {
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0),
                'prior': len(X_c) / len(X)
            }

    def _predict_gaussian(self, X):
        """
        Predict using Gaussian Naive Bayes.
        """
        posteriors = np.zeros((len(X), len(self.classes)))

        for idx, c in enumerate(self.classes):
            prior = np.log(self.statistics[c]['prior'])
            likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.statistics[c]['var']))
            likelihood -= 0.5 * np.sum(((X - self.statistics[c]['mean']) ** 2) / (self.statistics[c]['var']), axis=1)
            posteriors[:, idx] = prior + likelihood

        return np.array([self.classes[idx] for idx in np.argmax(posteriors, axis=1)])

    def get_params(self, deep=True):
        return {"method": self.method}


