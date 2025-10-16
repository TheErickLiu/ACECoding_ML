from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# 1. Load a dataset (e.g., Iris dataset)
iris = load_iris()
X = iris.data  # Features
y = iris.target # Target variable

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Create a KNN Classifier model
# n_neighbors specifies the 'k' value (number of neighbors)
# weights can be 'uniform' (default) or 'distance' for weighted KNN
knn_model = KNeighborsClassifier(n_neighbors=5, weights='uniform')

# 4. Train the model
knn_model.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = knn_model.predict(X_test)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of KNN model: {accuracy:.2f}")

# Example for KNN Regressor:
# from sklearn.neighbors import KNeighborsRegressor
# knn_regressor = KNeighborsRegressor(n_neighbors=5)
# knn_regressor.fit(X_train, y_train)
# y_pred_reg = knn_regressor.predict(X_test)

