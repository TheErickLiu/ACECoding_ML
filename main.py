import pandas as pd

df = pd.read_csv(‘train.csv’)
print(df.groupby(¨Sex¨)[¨Age¨].head())
# this is a panda program
print(df.groupby)


