import sklearn as sk
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("/Users/akshay/Advait/AceCoding/ACECoding_ML/train.csv")

# Calculate mean, median, and mode
mean = df.mean(numeric_only=True)["Age"]
class3 = df["Pclass"].value_counts()[3]  # Get the count of Class 3
# Create a new DataFrame sorted by Age in ascending order
ageorder = df.sort_values(by="Age", ascending=True)

# Print the results
print("Mean Age:\n", mean)
print("\nClass 3:\n", class3)
print("First 10 rows of the sorted DataFrame:\n", ageorder.head(10))
print("Dimensions of the DataFrame:", df.shape)
