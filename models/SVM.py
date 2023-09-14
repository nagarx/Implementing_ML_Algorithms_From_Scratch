import numpy as np
from cvxopt import matrix, solvers


class SVM:
    def __init__(self, kernel="linear", C=1.0, standardize=False, degree=3, sigma=1.0):
        self.mean = None
        self.std = None
        self.standardize = standardize
        self.kernel = kernel
        self.C = C
        self.alpha = None
        self.sv = None
        self.sv_y = None
        self.b = 0
        self.degree = degree
        self.sigma = sigma
        self.kernel_methods = {
            'linear': self._linear_kernel,
            'polynomial': self._polynomial_kernel,
            'rbf': self._rbf_kernel
        }

    def _standardize(self, X):
        """Standardize data."""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / std

    def _validate_input(self, X, y):
        """Validate input data."""
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("Input data and labels should be numpy arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y should be the same.")
        unique_labels = np.unique(y)
        if not np.array_equal(unique_labels, [-1, 1]):
            raise ValueError("Labels should be -1 or 1 for binary classification.")

    def _linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    def _polynomial_kernel(self, x1, x2):
        return (1 + np.dot(x1, x2)) ** self.degree

    def _rbf_kernel(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (self.sigma ** 2)))

    def decision_function(self, X):
        """
        Compute the decision function of the samples.
        Returns the signed distance of each sample to the hyperplane.
        """
        # If data was standardized during training, then standardize X as well
        if self.standardize:
            X = (X - self.mean) / self.std

        decision = np.dot(X, self.weights) + self.bias
        return decision

    def fit(self, X, y):
        # Validate input data
        self._validate_input(X, y)

        # Standardize data if required
        if self.standardize:
            X = self._standardize(X)
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)

        n_samples, n_features = X.shape

        # Compute the Gram matrix using the selected kernel
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel_methods[self.kernel](X[i], X[j])

        # Setup the Quadratic Programming problem
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n_samples))
        A = matrix(y, (1, n_samples), tc='d')
        b = matrix(0.0)
        G = matrix(np.vstack((np.eye(n_samples) * -1, np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))

        # Solve the QP problem
        solution = solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(solution['x'])

        # Extract support vectors
        sv = alpha > 1e-5
        ind = np.arange(len(alpha))[sv]
        self.alpha = alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        # Calculate the weights
        self.weights = np.sum(self.alpha[:, np.newaxis] * self.sv_y[:, np.newaxis] * self.sv, axis=0)

        # Compute the bias term
        self.bias = np.mean(self.sv_y - np.dot(self.sv, self.weights))

    def predict(self, X):
        if self.standardize:
            X = (X - self.mean) / self.std

        """Predict using the trained SVM model."""
        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                s += a * sv_y * self.kernel_methods[self.kernel](X[i], sv)
            y_pred[i] = s
        return np.sign(y_pred + self.b)

    def score(self, X, y):
        """Compute the accuracy of the SVM model."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def precision(self, X, y_true):
        """
        Compute the precision of the SVM model.
        Precision = TP / (TP + FP)
        """
        y_pred = self.predict(X)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == -1))
        return TP / (TP + FP) if (TP + FP) != 0 else 0

    def recall(self, X, y_true):
        """
        Compute the recall of the SVM model.
        Recall = TP / (TP + FN)
        """
        y_pred = self.predict(X)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FN = np.sum((y_pred == -1) & (y_true == 1))
        return TP / (TP + FN) if (TP + FN) != 0 else 0

    def f1_score(self, X, y_true):
        """
        Compute the F1-score of the SVM model.
        F1 = 2 * (precision * recall) / (precision + recall)
        """
        prec = self.precision(X, y_true)
        rec = self.recall(X, y_true)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0

    def metrics(self, X, y_true):
        """
        Compute all evaluation metrics for the SVM model.
        """
        return {
            'accuracy': self.score(X, y_true),
            'precision': self.precision(X, y_true),
            'recall': self.recall(X, y_true),
            'f1_score': self.f1_score(X, y_true)
        }


class MultiClassSVM:
    def __init__(self, standardize=False, kernel='linear', C=1.0, degree=3, sigma=1.0):
        self.standardize = standardize
        self.kernel = kernel
        self.C = C
        self.models = {}  # Store individual SVM models for each class here
        self.classes = None
        self.degree = degree
        self.sigma = sigma

    def fit(self, X, y):
        self.classes = np.unique(y)

        # Train an SVM for each class
        for c in self.classes:
            # Create binary labels for the current class vs. rest
            y_binary = np.where(y == c, 1, -1)

            # Initialize and train the SVM
            svm = SVM(standardize=self.standardize, kernel=self.kernel, C=self.C, degree=self.degree, sigma=self.sigma)
            svm.fit(X, y_binary)

            # Store the trained SVM model
            self.models[c] = svm

    def predict(self, X):
        # Get decision values for each SVM
        decision_values = {c: svm.decision_function(X) for c, svm in self.models.items()}

        # Predict the class with the highest decision value
        return np.array([max(decision_values, key=lambda k: decision_values[k][i]) for i in range(X.shape[0])])

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def precision(self, X, y, average='macro'):
        y_pred = self.predict(X)
        precisions = {}

        for c in self.classes:
            TP = np.sum((y_pred == c) & (y == c))
            FP = np.sum((y_pred == c) & (y != c))
            precisions[c] = TP / (TP + FP) if (TP + FP) != 0 else 0

        if average == 'macro':
            return np.mean(list(precisions.values()))
        elif average == 'micro':
            total_TP = sum([np.sum((y_pred == c) & (y == c)) for c in self.classes])
            total_FP = sum([np.sum((y_pred == c) & (y != c)) for c in self.classes])
            return total_TP / (total_TP + total_FP)
        elif average == 'weighted':
            supports = [np.sum(y == c) for c in self.classes]
            return np.average(list(precisions.values()), weights=supports)
        else:
            raise ValueError("Invalid average type. Choose from ['macro', 'micro', 'weighted']")

    def recall(self, X, y, average='macro'):
        y_pred = self.predict(X)
        recalls = {}

        for c in self.classes:
            TP = np.sum((y_pred == c) & (y == c))
            FN = np.sum((y_pred != c) & (y == c))
            recalls[c] = TP / (TP + FN) if (TP + FN) != 0 else 0

        if average == 'macro':
            return np.mean(list(recalls.values()))
        elif average == 'micro':
            total_TP = sum([np.sum((y_pred == c) & (y == c)) for c in self.classes])
            total_FN = sum([np.sum((y_pred != c) & (y == c)) for c in self.classes])
            return total_TP / (total_TP + total_FN)
        elif average == 'weighted':
            supports = [np.sum(y == c) for c in self.classes]
            return np.average(list(recalls.values()), weights=supports)
        else:
            raise ValueError("Invalid average type. Choose from ['macro', 'micro', 'weighted']")

    def f1_score(self, X, y, average='macro'):
        precision_val = self.precision(X, y, average)
        recall_val = self.recall(X, y, average)
        return 2 * (precision_val * recall_val) / (precision_val + recall_val) if (
                                                                                              precision_val + recall_val) != 0 else 0

    # Consolidated metrics method
    def metrics(self, X, y, average='macro'):
        return {
            'accuracy': self.score(X, y),
            'precision': self.precision(X, y, average),
            'recall': self.recall(X, y, average),
            'f1_score': self.f1_score(X, y, average)
        }



