
import numpy as np
from knn_from_scratch import KNN

# Sample data (2 features for simplicity)
X_train = np.array([[1, 1], [2, 2], [2, 3], [3, 3], [4, 5], [5, 5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

X_test = np.array([[1, 2], [2, 4], [4, 4], [4, 6]])
y_test = np.array([0, 0, 1, 1])

# Instantiate and fit the model
knn = KNN(k=3)
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)

# Calculate accuracy
accuracy = knn.score(X_test, y_test)

print("Predictions:", predictions)
print("Actual Labels:", y_test)
print("Accuracy:", accuracy)
