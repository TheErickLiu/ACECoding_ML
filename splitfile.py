from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = df.drop(columns=["Survived"])  # Assuming "Survived" is the target column
y = df["Survived"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Print the dimensions of the splits
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)