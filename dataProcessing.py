import pandas as pd

df = pd.read_parquet("yellow_tripdata_2009-01.parquet")

df.to_csv("tripdata200901.csv", index=False)

print(df)