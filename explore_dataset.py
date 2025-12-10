import pandas as pd

# 1. Read the prepared dataset
df = pd.read_csv("data/sea_activity_dataset.csv")

print("Shape (rows, columns):", df.shape)
print("\nColumns:", df.columns.tolist())

# 2. Descriptive statistics for feature columns
print("\nFeature statistics:")
print(df[["sigheight", "swellheight", "period", "windspeed", "winddirdegree"]].describe())

# 3. Class distribution for bulk
print("\nBulk value counts (absolute):")
print(df["bulk"].value_counts())

print("\nBulk value counts (relative):")
print(df["bulk"].value_counts(normalize=True))