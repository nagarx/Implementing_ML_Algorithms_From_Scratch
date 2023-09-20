
import numpy as np

class WeakClassifier:
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions

class AdaBoost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))
        self.clfs = []

        for _ in range(self.n_clf):
            clf = WeakClassifier()
            min_error = float('inf')

            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    for polarity in [1, -1]:
                        predictions = np.ones(n_samples)
                        predictions[polarity * X_column < polarity * threshold] = -1

                        error = sum(w[y != predictions])

                        if error < min_error:
                            clf.polarity = polarity
                            clf.threshold = threshold
                            clf.feature_index = feature_i
                            min_error = error

            clf.alpha = 0.5 * np.log((1 - min_error) / (min_error + 1e-10))
            w *= np.exp(-clf.alpha * y * self._predict(X, clf))
            w /= np.sum(w)
            self.clfs.append(clf)

    def _predict(self, X, clf):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
        predictions[negative_idx] = -1
        return predictions

    def predict(self, X):
        clf_preds = [clf.alpha * self._predict(X, clf) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred
    