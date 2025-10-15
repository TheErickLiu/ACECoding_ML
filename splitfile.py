#from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
#X = df.drop(columns=["Survived"])  # Assuming "Survived" is the target column
#y = df["Survived"]

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Print the dimensions of the splits
#print("X_train shape:", X_train.shape)
#print("X_test shape:", X_test.shape)
#print("y_train shape:", y_train.shape)
#print("y_test shape:", y_test.shape)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Load the dataset
df = pd.read_csv("/Users/akshay/Advait/AceCoding/ACECoding_ML/train.csv")

# Define features (X) and target (y)
X = df.drop(columns=["Survived"])  # Assuming "Survived" is the target column
y = df["Survived"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize the KNN model with a chosen number of neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the KNN model on the training data
knn_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn_model.predict(X_test)

# Print predictions
print("Predictions on test data:", y_pred)