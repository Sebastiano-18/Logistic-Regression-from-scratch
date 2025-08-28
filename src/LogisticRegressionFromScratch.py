import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
  def __init__(self, learning_rate=0.01, n_iterations=1000):
    self.learning_rate=learning_rate
    self.n_iterations=n_iterations
    self.weights = None
    self.bias = None

  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))

  def fit(self, X, y):
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0

    # Gradient Discent
    for _ in range(self.n_iterations):
      linear_model = np.dot(X, self.weights) + self.bias
      y_predicted = self.sigmoid(linear_model)

      # Gradients
      dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
      db = (1 / n_samples) * np.sum(y_predicted - y)

      # Update parameters
      self.weights -= self.learning_rate * dw
      self.bias -= self.learning_rate * db

  def predict(self, X):
    linear_model = np.dot(X, self.weights) + self.bias
    y_predicted = self.sigmoid(linear_model)
    return np.where(y_predicted >= 0.5, 1, 0)

if __name__ == "__main__":
  # Generate binary classification dataset
  X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

  # Splitting the dataset
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Standardize the Features
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # Train the Logistic Regression From scratch
  clf = LogisticRegression(learning_rate=0.1, n_iterations=1000)
  clf.fit(X_train, y_train)
  predictions = clf.predict(X_test)

  accuracy = np.mean(predictions == y_test)
  print(f"Logistic Regression Accuracy:  {accuracy:.2f}")