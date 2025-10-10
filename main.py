

#print(df.groupby["Age"].head())
# this is a panda program
# print(df.groupby)
import pandas as pd

# Load the dataset
df = pd.read_csv("train.csv")

# Calculate mean, median, and mode
mean = df.mean(numeric_only=True)["Age"]
class3 = df["Pclass"].value_counts()[3]  # Get the count of Class 3
#median = df.median(numeric_only=True)
#mode = df.mode(numeric_only=True).iloc[0]  # Take the first row for mode

# Print the results
print("Mean:\n", mean)
print("\nClass 3:\n", class3)
#print("\nMedian:\n", median)
#print("\nMode:\n", mode)
