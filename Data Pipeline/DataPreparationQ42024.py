import dask.dataframe as dd
import pandas as pd
import holidays

# Load Q4 2024 trip data across all vehicle types using Dask for out-of-core processing
df_fhvhv = dd.read_parquet(["drive/MyDrive/BT_Florian_2026/ParquetFiles/fhvhv_tripdata_2024-10.parquet",
                            "drive/MyDrive/BT_Florian_2026/ParquetFiles/fhvhv_tripdata_2024-11.parquet",
                            "drive/MyDrive/BT_Florian_2026/ParquetFiles/fhvhv_tripdata_2024-12.parquet"])
df_yellow = dd.read_parquet(["drive/MyDrive/BT_Florian_2026/ParquetFiles/yellow_tripdata_2024-10.parquet",
                              "drive/MyDrive/BT_Florian_2026/ParquetFiles/yellow_tripdata_2024-11.parquet",
                              "drive/MyDrive/BT_Florian_2026/ParquetFiles/yellow_tripdata_2024-12.parquet"])
df_green = dd.read_parquet(["drive/MyDrive/BT_Florian_2026/ParquetFiles/green_tripdata_2024-10.parquet",
                              "drive/MyDrive/BT_Florian_2026/ParquetFiles/green_tripdata_2024-11.parquet",
                              "drive/MyDrive/BT_Florian_2026/ParquetFiles/green_tripdata_2024-12.parquet"])

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
df_2024Q4 = dd.concat([df_yellow,df_green,df_fhvhv])


print("Success_1")

# Sort the combined dataset chronologically
df_2024Q4 = df_2024Q4.sort_values("PUDatetime")

# Extract temporal features for modeling (day of the week and NY state holidays)
df_2024Q4["DayOfWeek"] = df_2024Q4["PUDatetime"].dt.dayofweek

# Note: Renamed variable to 'ny_holidays' to prevent shadowing the imported 'holidays' module
ny_holidays = holidays.US(subdiv="NY", years=2024)
holiday_dates = pd.to_datetime(list(ny_holidays.keys()))

PUDates = df_2024Q4["PUDatetime"].dt.floor("D")
df_2024Q4["is_holiday"] = PUDates.isin(holiday_dates)

# Export the processed dataset and save a small CSV sample for quick visual inspection
df_2024Q4.to_parquet("drive/MyDrive/BT_Florian_2026/2024Q4_tripdata.parquet")
df_2024Q4.head(1000).to_csv("drive/MyDrive/BT_Florian_2026/2024Q4_tripdata.csv")

print("Success_2")