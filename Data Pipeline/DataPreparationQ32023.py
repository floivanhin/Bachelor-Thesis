import dask.dataframe as dd
import pandas as pd
import holidays

df_fhvhv = dd.read_parquet(["drive/MyDrive/BT_Florian_2026/ParquetFiles/fhvhv_tripdata_2023-07.parquet",
                            "drive/MyDrive/BT_Florian_2026/ParquetFiles/fhvhv_tripdata_2023-08.parquet",
                            "drive/MyDrive/BT_Florian_2026/ParquetFiles/fhvhv_tripdata_2023-09.parquet"])
df_yellow = dd.read_parquet(["drive/MyDrive/BT_Florian_2026/ParquetFiles/yellow_tripdata_2023-07.parquet",
                              "drive/MyDrive/BT_Florian_2026/ParquetFiles/yellow_tripdata_2023-08.parquet",
                              "drive/MyDrive/BT_Florian_2026/ParquetFiles/yellow_tripdata_2023-09.parquet"])
df_green = dd.read_parquet(["drive/MyDrive/BT_Florian_2026/ParquetFiles/green_tripdata_2023-07.parquet",
                              "drive/MyDrive/BT_Florian_2026/ParquetFiles/green_tripdata_2023-08.parquet",
                              "drive/MyDrive/BT_Florian_2026/ParquetFiles/green_tripdata_2023-09.parquet"])


df_fhvhv = df_fhvhv.rename(columns={"pickup_datetime":"PUDatetime", "dropoff_datetime":"DODatetime", "airport_fee":"Airport_fee",
                                    "trip_miles":"trip_distance", "tolls":"tolls_amount", "tips":"tip_amount",
                                    "base_passenger_fare":"fare_amount", "sales_tax":"tax"})
df_fhvhv["taxi_type"] = "for-hire vehicle"

fare_cols = ["fare_amount","tolls_amount","bcf","tax","congestion_surcharge","Airport_fee","tip_amount"]
df_fhvhv["total_amount"] = df_fhvhv[fare_cols].fillna(0).sum(axis=1)

df_green = df_green.rename(columns={"lpep_pickup_datetime":"PUDatetime", "lpep_dropoff_datetime":"DODatetime", "mta_tax":"tax"})
df_green["taxi_type"] = "green_taxi"
df_yellow = df_yellow.rename(columns={"tpep_pickup_datetime":"PUDatetime", "tpep_dropoff_datetime":"DODatetime", "mta_tax": "tax"})
df_yellow["taxi_type"] = "yellow_taxi"

df_2023Q3 = dd.concat([df_yellow,df_green,df_fhvhv])

'''
df_2023Q3 = df_2023Q3[
    (df_2023Q3["PUDatetime"] >= "2023-07-01 00:00:00") &    # Can I just remove them? 
    (df_2023Q3["PUDatetime"] < "2023-10-01 00:00:00")       # These trips probably happened in my timeframe, they are just faulty
]

df_2023Q3 = df_2023Q3[
    (df_2023Q3["trip_distance"] > 0) &
    (df_2023Q3["trip_distance"] < 200)          # How do I define an appropriate upper limit without it being arbitrary? Also same as the above
]
'''

print("Success_1")

df_2023Q3 = df_2023Q3.sort_values("PUDatetime")

df_2023Q3["DayOfWeek"] = df_2023Q3["PUDatetime"].dt.dayofweek

holidays = holidays.US(subdiv="NY", years=2023)
holiday_dates = pd.to_datetime(list(holidays.keys()))

PUDates = df_2023Q3["PUDatetime"].dt.floor("D")

df_2023Q3["is_holiday"] = PUDates.isin(holiday_dates)


df_2023Q3.to_parquet("drive/MyDrive/BT_Florian_2026/2023Q3_tripdata.parquet")
df_2023Q3.head(1000).to_csv("drive/MyDrive/BT_Florian_2026/2023Q3_tripdata.csv")

print("Success_2")