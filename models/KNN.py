
import numpy as np
from heapq import heapify, heappush, heappop
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        """
        Store the training data to be used during prediction.
        """
        self.X_train = X_train
        self.y_train = y_train
    
    def _euclidean_distance(self, point1, point2):
        """
        Calculate the Euclidean distance between two points.
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def _get_neighbors(self, X_test):
        """
        Get the k-nearest neighbors for a given test point.
        """
        distances_heap = []
        heapify(distances_heap)
        
        for i in range(len(self.X_train)):
            distance = self._euclidean_distance(self.X_train[i], X_test)
            heappush(distances_heap, (distance, self.y_train[i]))
        
        neighbors = [heappop(distances_heap)[1] for _ in range(self.k)]
        return neighbors
    
    def predict(self, X_test):
        """
        Predict the class label for each test point.
        """
        predictions = []
        for x in X_test:
            neighbors = self._get_neighbors(x)
            most_common = Counter(neighbors).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)
    
    def score(self, X_test, y_test):
        """
        Calculate the accuracy of the model on test data.
        """
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy
