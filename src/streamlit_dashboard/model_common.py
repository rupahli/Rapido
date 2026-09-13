from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parents[1] / "DataCSVFiles"
MODEL_DIR = Path(__file__).resolve().parent / "model_artifacts"


def load_sources():
    bookings = pd.read_csv(DATA_DIR / "bookings.csv")
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    drivers = pd.read_csv(DATA_DIR / "drivers.csv")
    locations = pd.read_csv(DATA_DIR / "location_demand.csv")
    return bookings, customers, drivers, locations


def prepare_frame(bookings, customers, drivers):
    frame = bookings.copy()
    frame["booking_date"] = pd.to_datetime(frame["booking_date"])
    frame["booking_time"] = pd.to_datetime(frame["booking_time"])
    frame["booking_hour"] = frame["booking_time"].dt.hour
    frame["booking_month"] = frame["booking_date"].dt.month
    frame["is_peak_time"] = frame["booking_hour"].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)
    frame["fare_per_km"] = frame["booking_value"] / frame["ride_distance_km"].clip(lower=0.01)
    frame["fare_markup"] = frame["booking_value"] / frame["base_fare"].clip(lower=0.01)
    frame["surge_cost"] = frame["surge_multiplier"] - 1
    frame = frame.merge(customers, on="customer_id", how="left", validate="many_to_one")
    return frame.merge(
        drivers,
        on="driver_id",
        how="left",
        suffixes=("", "_driver"),
        validate="many_to_one",
    )


def make_preprocessor(frame):
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    categorical = frame.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


LEAKAGE_COLUMNS = [
    "booking_status",
    "actual_ride_time_min",
    "incomplete_ride_reason",
    "booking_id",
    "customer_id",
    "driver_id",
    "booking_date",
    "booking_time",
    "pickup_location",
    "drop_location",
]
