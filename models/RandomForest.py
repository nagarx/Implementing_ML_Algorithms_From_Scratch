
import numpy as np

# Decision Tree Stump class
class DecisionTreeStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None
    
    def fit(self, X, y):
        n_features = X.shape[1]
        best_gini = float('inf')
        
        for feature_index in range(n_features):
            possible_thresholds = np.unique(X[:, feature_index])
            for threshold in possible_thresholds:
                left_mask = X[:, feature_index] < threshold
                right_mask = ~left_mask
                
                left_gini = self._calculate_gini(y[left_mask])
                right_gini = self._calculate_gini(y[right_mask])
                
                gini = left_gini + right_gini
                
                if gini < best_gini:
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.left_value = np.mean(y[left_mask])
                    self.right_value = np.mean(y[right_mask])
                    best_gini = gini
                    
    def _calculate_gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini * len(y)
    
    def predict(self, X):
        mask = X[:, self.feature_index] < self.threshold
        predictions = np.ones(X.shape[0]) * self.right_value
        predictions[mask] = self.left_value
        return predictions

# Random Forest class
class RandomForest:
    def __init__(self, n_trees=10):
        self.n_trees = n_trees
        self.trees = []
        
    def add_tree(self, tree):
        self.trees.append(tree)
        
    def fit(self, X, y):
        for _ in range(self.n_trees):
            X_boot, y_boot = self.balanced_bootstrap_sample(X, y)
            tree_stump = DecisionTreeStump()
            tree_stump.fit(X_boot, y_boot)
            self.add_tree(tree_stump)
    
    def predict(self, X):
        predictions = np.zeros((self.n_trees, X.shape[0]))
        for i, tree in enumerate(self.trees):
            predictions[i] = tree.predict(X)
        combined_predictions = np.mean(predictions, axis=0)
        return np.round(combined_predictions)
    
    def balanced_bootstrap_sample(self, X, y):
        classes = np.unique(y)
        n_samples_per_class = len(y) // len(classes)
        boot_X, boot_y = [], []
        for cls in classes:
            X_cls = X[y == cls]
            y_cls = y[y == cls]
            indices = np.random.choice(len(y_cls), n_samples_per_class, replace=True)
            boot_X.append(X_cls[indices])
            boot_y.append(y_cls[indices])
        boot_X = np.concatenate(boot_X)
        boot_y = np.concatenate(boot_y)
        return boot_X, boot_y
