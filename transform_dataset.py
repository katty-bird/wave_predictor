import pandas as pd

# 1. Read raw CSV files
day = pd.read_csv("data/day_forecast.csv", sep=";")
hour = pd.read_csv("data/hour_forecast.csv", sep=";")
cond = pd.read_csv("data/sea_condition_fact.csv", sep=";")

# 2. Keep only the columns from day_forecast that are needed for joining
day_small = day[["iddayforecast", "date"]].copy()

# 3. Attach the date information to the hourly forecast
hour_merged = hour.merge(day_small, on="iddayforecast", how="left")

# 4. Build a proper datetime column from date + time (0, 100, 2300 → HH:MM)
time_str = hour_merged["time"].astype(str).str.zfill(4)
hh = time_str.str.slice(0, 2)
mm = time_str.str.slice(2, 4)

datetime_str = hour_merged["date"] + " " + hh + ":" + mm + ":00"
hour_merged["datetime"] = pd.to_datetime(datetime_str)

# 5. Convert the date in sea_condition_fact to datetime as well
cond["datetime"] = pd.to_datetime(cond["date"])

# 6. Merge forecast data with sea condition labels by datetime
# (if needed later, idspot can be added to this merge as well)
merged = hour_merged.merge(
    cond[["bulk", "texture", "shape", "uniformity", "score", "datetime"]],
    on="datetime",
    how="inner",
)

print("Merged shape:", merged.shape)
print("Sample rows:")
print(merged.head())

# 7. Select only the feature columns + target, and drop missing values
feature_cols = ["sigheight", "swellheight", "period", "windspeed", "winddirdegree"]
target_col = "bulk"

dataset = merged[feature_cols + [target_col]].dropna()

print("\nDataset shape (after column selection & dropna):", dataset.shape)
print("Bulk value counts:")
print(dataset["bulk"].value_counts())

# 8. Save the cleaned dataset for model training
dataset.to_csv("data/sea_activity_dataset.csv", index=False)
print("\nSaved cleaned dataset to data/sea_activity_dataset.csv")