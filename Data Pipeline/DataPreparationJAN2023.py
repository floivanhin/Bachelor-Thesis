import dask.dataframe as dd
import pandas as pd
import holidays

# Load January 2023 trip data across all vehicle types using Dask
df_fhvhv = dd.read_parquet("drive/MyDrive/BT_Florian_2026/ParquetFiles/fhvhv_tripdata_2023-01.parquet")
df_yellow = dd.read_parquet("drive/MyDrive/BT_Florian_2026/ParquetFiles/yellow_tripdata_2023-01.parquet")
df_green = dd.read_parquet("drive/MyDrive/BT_Florian_2026/ParquetFiles/green_tripdata_2023-01.parquet")

# Standardize column names across datasets to enable seamless vertical concatenation
df_fhvhv = df_fhvhv.rename(columns={"pickup_datetime":"PUDatetime", "dropoff_datetime":"DODatetime", "airport_fee":"Airport_fee",
                                    "trip_miles":"trip_distance", "tolls":"tolls_amount", "tips":"tip_amount",
                                    "base_passenger_fare":"fare_amount", "sales_tax":"tax"})
df_fhvhv["taxi_type"] = "for-hire vehicle"

# FHV data lacks a pre-aggregated total cost; calculate it by summing individual fare components
fare_cols = ["fare_amount","tolls_amount","bcf","tax","congestion_surcharge","Airport_fee","tip_amount"]
df_fhvhv["total_amount"] = df_fhvhv[fare_cols].fillna(0).sum(axis=1)

df_green = df_green.rename(columns={"lpep_pickup_datetime":"PUDatetime", "lpep_dropoff_datetime":"DODatetime", "mta_tax":"tax"})
df_green["taxi_type"] = "green_taxi"
df_yellow = df_yellow.rename(columns={"tpep_pickup_datetime":"PUDatetime", "tpep_dropoff_datetime":"DODatetime", "mta_tax": "tax"})
df_yellow["taxi_type"] = "yellow_taxi"

# Merge all standardized vehicle datasets into a single unified DataFrame
df_2023Q1 = dd.concat([df_yellow,df_green,df_fhvhv])

print("Success_1")

# Sort the combined dataset chronologically
df_2023Q1 = df_2023Q1.sort_values("PUDatetime")

# Extract temporal features for modeling
df_2023Q1["DayOfWeek"] = df_2023Q1["PUDatetime"].dt.dayofweek

# Note: Renamed variable to 'ny_holidays' to prevent shadowing the imported 'holidays' module
ny_holidays = holidays.US(subdiv="NY", years=2023)
holiday_dates = pd.to_datetime(list(ny_holidays.keys()))

PUDates = df_2023Q1["PUDatetime"].dt.floor("D")

df_2023Q1["is_holiday"] = PUDates.isin(holiday_dates)

# Export the processed dataset and save a small CSV sample for quick visual inspection
df_2023Q1.to_parquet("drive/MyDrive/BT_Florian_2026/2023JAN_tripdata.parquet")
df_2023Q1.head(1000).to_csv("drive/MyDrive/BT_Florian_2026/2023JAN_tripdata.csv")

print("Success_2")