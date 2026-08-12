import pandas as pd
import numpy as np
import optuna
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import time
import kagglehub

path = kagglehub.dataset_download("florianhinrichsen/nyc-taxi-data-2023-24-normalized/version/3", force_download=True)

datasets = {f"{path}/hourlyDOs202324_Normalized.parquet": "df_DOs",
            f"{path}/hourlyDOs202324WEATHER_Normalized.parquet":"df_DOsWEATHER",
            f"{path}/hourlyDOs202324CYCLt_Normalized.parquet": "df_DOsCYCLt",
            f"{path}/hourlyDOs202324WEATHER_CYCLt_Normalized.parquet":"df_DOsWEATHER_CYCLt",
            f"{path}/hourlyPUs202324_Normalized.parquet":"df_PUs",
            f"{path}/hourlyPUs202324WEATHER_Normalized.parquet":"df_PUsWEATHER",
            f"{path}/hourlyPUs202324CYCLt_Normalized.parquet":"df_PUsCYCLt",
            f"{path}/hourlyPUs202324WEATHER_CYCLt_Normalized.parquet":"df_PUsWEATHER_CYCLt"}


first_block_end = pd.to_datetime("2024-01-01")
test_set_start = pd.to_datetime("2024-07-01")

DataBase_URL = "sqlite:///Ridge_Regression_tuning.db"

for file in datasets:
    dataset_name = datasets[file]
    df = pd.read_parquet(file)
    df = df.rename(columns={df.columns[1]:"date"})
    df["year"] = df["year"].map({2023: 0, 2024: 1})
    df = df.drop(columns=["DOLocationID", "PULocationID"], errors="ignore")
    df = df.drop(columns=["precipitation"], errors='ignore')
    df = df.dropna()

    df_test = df[df["date"]>=test_set_start].copy()
    df_dev = df[df["date"]<test_set_start].copy()

    X_test = df_test.drop(columns=["trip_count","date"])
    y_test = df_test["trip_count"]

    X_dev = df_dev.drop(columns=["trip_count","date"])
    y_dev = df_dev["trip_count"]

    custom_folds = []
    for month in range(6):
        val_month_start = first_block_end + pd.DateOffset(months=month)
        val_month_end = val_month_start + pd.DateOffset(months=1)

        train_end_i = df_dev["date"].searchsorted(val_month_start)
        val_end_i = df_dev["date"].searchsorted(val_month_end)

        train_indices = np.arange(0,train_end_i)
        val_indices = np.arange(train_end_i,val_end_i)

        custom_folds.append((train_indices,val_indices))

    def objective(trial):
        alpha = trial.suggest_float("alpha",0.001,10000,log=True)

        model = Ridge(alpha=alpha)

        fold_errors = []

        for train_i, val_i in custom_folds:
            X_train, X_val = X_dev.iloc[train_i], X_dev.iloc[val_i]
            y_train, y_val = y_dev.iloc[train_i], y_dev.iloc[val_i]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_val)

            mae = mean_absolute_error(y_val,pred)
            fold_errors.append(mae)

        return np.mean(fold_errors)
        

    study = optuna.create_study(
        study_name=f"ridge_{dataset_name}",
        storage=DataBase_URL,
        load_if_exists=True,
        direction="minimize"
    )

    tuning_start = time.perf_counter()
    study.optimize(objective, n_trials=100)
    tuning_time = time.perf_counter() - tuning_start

    best_alpha = study.best_params["alpha"]

    print(f"Tuning hyperparameters for {dataset_name}:{best_alpha}")

    final_model = Ridge(alpha=best_alpha)
    training_start = time.perf_counter()
    final_model.fit(X_dev, y_dev)
    training_time = time.perf_counter() - training_start

    coefficients = final_model.coef_
    coefficients_df = pd.DataFrame({
        "Feature": X_dev.columns,
        "Coefficient": coefficients
    })

    print(coefficients_df.to_string(index=False))

    testing_start = time.perf_counter()
    test_preds = final_model.predict(X_test)
    final_test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    final_test_mae = mean_absolute_error(y_test,test_preds)
    testing_time = time.perf_counter() - testing_start

    print(f"Overview for Dataset {dataset_name}:")

    print(f">>> TEST SET PERFORMANCE (RMSE): {final_test_rmse:.4f} <<<")
    print(f">>> TEST SET PERFORMANCE (MAE): {final_test_mae:.4f} <<<")

    print(f"  • Tuning Time (100 trials): {tuning_time:.2f} seconds")
    print(f"  • Final Training Time:      {training_time:.4f} seconds")
    print(f"  • Testing Time:             {testing_time:.4f} seconds")
