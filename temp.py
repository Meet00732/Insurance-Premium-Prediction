import pandas as pd

df = pd.read_csv("insurance.csv")

# print(df['region'].unique())
print(df['sex'].unique())