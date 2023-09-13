import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000, l1_lambda=0.1, l2_lambda=0.1, early_stopping=True,
                 tol=1e-4):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.early_stopping = early_stopping
        self.tol = tol
        self.weights, self.bias = None, None
        self.losses = []

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y_true, y_pred):
        n_samples = len(y_true)
        epsilon = 1e-15  # To prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Cross-entropy loss
        loss = - (1 / n_samples) * (np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

        # L1 regularization
        l1_loss = self.l1_lambda * np.sum(np.abs(self.weights))

        # L2 regularization
        l2_loss = self.l2_lambda * np.sum(self.weights ** 2)

        return loss + l1_loss + l2_loss

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape

        # 1. Initialization
        self.weights = np.zeros(n_features)
        self.bias = 0
        best_weights, best_bias = self.weights, self.bias
        min_val_loss = float('inf')

        # 4. Gradient Descent
        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y)) + self.l1_lambda * np.sign(
                self.weights) + self.l2_lambda * self.weights
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update parameters
            self.weights -= self.learning_rate / (1 + 0.01 * i)  # Learning rate decay
            self.bias -= self.learning_rate * db

            # Compute loss
            self.losses.append(self._compute_loss(y, predictions))

            # Early stopping
            if self.early_stopping and X_val is not None and y_val is not None:
                val_preds = self._sigmoid(np.dot(X_val, self.weights) + self.bias)
                val_loss = self._compute_loss(y_val, val_preds)
                if val_loss + self.tol < min_val_loss:
                    min_val_loss = val_loss
                    best_weights, best_bias = self.weights, self.bias
                elif val_loss > min_val_loss:
                    self.weights, self.bias = best_weights, best_bias
                    break

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(linear_model)
        prediction_class = [1 if i > 0.5 else 0 for i in predictions]
        return prediction_class

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)