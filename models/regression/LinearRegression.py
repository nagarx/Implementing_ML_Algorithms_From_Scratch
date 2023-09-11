
import numpy as np

class LinearRegression:
    """
        Implementation of a stable linear regression model with optional L1 (Lasso) and L2 (Ridge) regularization.
    """

    def __init__(self, alpha=0.0, regularization=None, learning_rate=0.01, iterations=1000):
        """
                Initialize the linear regression model with specified hyperparameters.

                Parameters:
                - alpha: Regularization strength.
                - regularization: Type of regularization ('l1' for Lasso, 'l2' for Ridge, or None for no regularization).
                - learning_rate: Learning rate for optimization algorithms (currently used only for future implementations).
                - iterations: Number of iterations for optimization algorithms (currently used only for future implementations).
        """

        self.coefficients = None
        self.alpha = alpha
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.iterations = iterations

    def _validate_input(self, X, y):
        """
                Validate the input data to ensure consistency and proper shape.

                Parameters:
                - X: Feature matrix.
                - y: Target vector.

                Raises:
                - ValueError: If input data is inconsistent or has incorrect dimensions.
        """

        if X.size == 0 or y.size == 0:
            raise ValueError("Input data cannot be empty.")
        if len(X) != len(y):
            raise ValueError("The dimensions of X and y do not match.")

    def _coordinate_descent_lasso(self, X, y):
        """
                Apply coordinate gradient descent for Lasso regularization.

                Parameters:
                - X: Normalized feature matrix with an added bias column.
                - y: Target vector.

                Returns:
                - coefficients: Coefficients determined after applying Lasso regularization.
        """

        m, n = X.shape
        coefficients = np.zeros(n)
        for _ in range(self.iterations):
            for j in range(n):
                tmp_coef = coefficients.copy()
                tmp_coef[j] = 0.0
                residual = y - X @ tmp_coef

                if j == 0:  # Intercept
                    coefficients[j] = np.sum(residual) / m
                else:
                    rho = X[:, j] @ residual
                    if rho < -self.alpha / 2:
                        coefficients[j] = (rho + self.alpha / 2) / m
                    elif rho > self.alpha / 2:
                        coefficients[j] = (rho - self.alpha / 2) / m
                    else:
                        coefficients[j] = 0.0
        return coefficients

    def fit(self, X, y):
        """
                Train the linear regression model using the provided data.

                Parameters:
                - X: Feature matrix.
                - y: Target vector.
        """

        self._validate_input(X, y)

        # Convert X and y to numpy arrays
        X, y = np.array(X), np.array(y)

        # Normalize features
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        # If X is 1D (simple linear regression), reshape it to 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Add a column of ones to X for the intercept coefficient
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        if self.regularization == "l2":  # Ridge
            regularization_term = self.alpha * np.eye(X.shape[1])
            regularization_term[0, 0] = 0  # Don't regularize the intercept
            self.coefficients = np.linalg.inv(X.T @ X + regularization_term) @ X.T @ y
        elif self.regularization == "l1":  # Lasso
            self.coefficients = self._coordinate_descent_lasso(X, y)
        else:  # No regularization or gradient descent
            self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        """
                Predict the target values using the trained model.

                Parameters:
                - X: Feature matrix.

                Returns:
                - predictions: Predicted values for the input feature matrix.
        """

        X = np.array(X)

        # Normalize features
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        # If X is 1D, reshape it to 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Add a column of ones to X for the intercept
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        return X @ self.coefficients

    def score(self, X, y):
        """
                Calculate the R^2 score of the model using the provided data.

                Parameters:
                - X: Feature matrix.
                - y: Target vector.

                Returns:
                - r2: R^2 score of the model's predictions.
        """

        y_pred = self.predict(X)
        y = np.array(y)

        y_mean = np.mean(y)
        ss_total = np.sum((y - y_mean) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)

        r2 = 1 - (ss_residual / ss_total)
        return r2

    def metrics(self, X, y):
        """
                Compute various evaluation metrics for the model's predictions.

                Parameters:
                - X: Feature matrix.
                - y: Target vector.

                Returns:
                - metrics_dict: A dictionary containing MAE, MSE, and RMSE values.
        """

        y_pred = self.predict(X)
        y = np.array(y)

        mae = np.mean(np.abs(y - y_pred))
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)

        return {"MAE": mae, "MSE": mse, "RMSE": rmse}

    def __str__(self):
        """
                Provide a string representation of the model, showing the trained coefficients.

                Returns:
                - str_repr: String representation of the model's coefficients.
        """

        equation = "y = "
        if self.coefficients is not None:
            equation += f"{self.coefficients[0]:.3f} "
            for i, coef in enumerate(self.coefficients[1:]):
                equation += f"+ {coef:.3f} * x{i+1} "
        else:
            equation = "Model has not been trained yet."
        return equation