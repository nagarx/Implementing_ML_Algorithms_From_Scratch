import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """Build decision tree classifier."""
        self.classes_ = np.unique(y)
        self.class2index_ = {c: idx for idx, c in enumerate(self.classes_)}
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        """Predict class for X."""
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        """Predict class for a single sample."""
        node = self.tree
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return self.classes_[node.predicted_class]

    # Helper functions
    def _entropy(self, y):
        m = len(y)
        return -sum([np.sum(y == c) / m * np.log2(np.sum(y == c) / m) for c in self.classes_])

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in self.classes_]
        best_gini = 1.0 - sum((num / m) ** 2 for num in num_parent)
        best_idx, best_thr = None, None

        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = self.class2index_[classes[i - 1]]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_))
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_))
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _split(self, X, y, feature_index, threshold):
        left_idx = np.where(X[:, feature_index] < threshold)
        right_idx = np.where(X[:, feature_index] >= threshold)
        return X[left_idx], X[right_idx], y[left_idx], y[right_idx]

    def _build_tree(self, X, y, depth=0):
        """Build a decision tree recursively."""
        num_samples_per_class = [np.sum(y == c) for c in self.classes_]
        predicted_class = np.argmax(num_samples_per_class)
        node = self.Node(
            gini=1 - sum((np.sum(y == c) / len(y)) ** 2 for c in self.classes_),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left, indices_right, y_left, y_right = self._split(X, y, idx, thr)
                node.feature_index = idx
                node.threshold = thr
                node.left = self._build_tree(indices_left, y_left, depth + 1)
                node.right = self._build_tree(indices_right, y_right, depth + 1)
        return node

    class Node:
        def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
            self.gini = gini
            self.num_samples = num_samples
            self.num_samples_per_class = num_samples_per_class
            self.predicted_class = predicted_class
            self.feature_index = 0
            self.threshold = 0
            self.left = None
            self.right = None
