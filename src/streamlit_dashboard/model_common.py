from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
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


# customers.csv / drivers.csv are aggregated over the WHOLE bookings.csv file, including
# each booking's own outcome (verified: customers.csv.total_bookings matches each
# customer's row count in bookings.csv at 100%, cancelled_rides matches at 100%, and the
# same holds for drivers.csv). Using those columns as-is during training leaks a booking's
# own outcome into its own features. These are replaced by point-in-time equivalents
# rebuilt from bookings.csv (see add_point_in_time_history) that only look at strictly
# earlier bookings.
CUSTOMER_RAW_HISTORY_COLUMNS = [
    "total_bookings",
    "completed_rides",
    "cancelled_rides",
    "incomplete_rides",
    "cancellation_rate",
    "customer_cancel_flag",  # deterministic threshold on cancellation_rate (>0.20)
]

# delay_rate, delay_count, and avg_pickup_delay_min define driver_delay_flag almost by
# construction (driver_delay_flag == 1 exactly when delay_rate > 0.10 in this dataset), so
# they cannot be used as predictors for it. total_assigned_rides is replaced by the
# point-in-time driver_prior_bookings for the same reason as the customer columns above.
DRIVER_RAW_LEAK_COLUMNS = [
    "delay_rate",
    "delay_count",
    "avg_pickup_delay_min",
    "total_assigned_rides",
    "incomplete_rides",  # whole-history count; replaced by driver_prior_incomplete
]

HISTORY_SMOOTHING = 5


def _smoothed_rate(positive, total, global_rate, smoothing=HISTORY_SMOOTHING):
    return (positive + smoothing * global_rate) / (total + smoothing)


def add_point_in_time_history(bookings):
    """Rebuild customer/driver history counts as strictly-prior, point-in-time features.

    Sorts bookings chronologically and, per customer/driver, computes cumulative counts
    that exclude the current row, so a booking's own outcome never leaks into its own
    features (unlike the raw customers.csv/drivers.csv aggregates it replaces).
    """
    df = bookings.copy()
    order = pd.to_datetime(df["booking_date"]) + pd.to_timedelta(
        pd.to_datetime(df["booking_time"], format="%H:%M:%S").dt.strftime("%H:%M:%S")
    )
    df = df.assign(_order=order).sort_values("_order").reset_index(drop=True)

    is_cancelled = (df["booking_status"] == "Cancelled").astype(int)
    is_completed = (df["booking_status"] == "Completed").astype(int)
    is_incomplete = (df["booking_status"] == "Incomplete").astype(int)
    global_cancel_rate = is_cancelled.mean()
    global_incomplete_rate = is_incomplete.mean()

    df["customer_prior_bookings"] = df.groupby("customer_id").cumcount()
    df["customer_prior_cancelled"] = is_cancelled.groupby(df["customer_id"]).cumsum() - is_cancelled
    df["customer_prior_completed"] = is_completed.groupby(df["customer_id"]).cumsum() - is_completed
    df["customer_prior_incomplete"] = is_incomplete.groupby(df["customer_id"]).cumsum() - is_incomplete
    df["customer_prior_cancel_rate"] = _smoothed_rate(
        df["customer_prior_cancelled"], df["customer_prior_bookings"], global_cancel_rate
    )

    df["driver_prior_bookings"] = df.groupby("driver_id").cumcount()
    df["driver_prior_incomplete"] = is_incomplete.groupby(df["driver_id"]).cumsum() - is_incomplete
    df["driver_prior_incomplete_rate"] = _smoothed_rate(
        df["driver_prior_incomplete"], df["driver_prior_bookings"], global_incomplete_rate
    )

    # Route popularity ("City_Pair" in the project brief). pickup_location/drop_location
    # only have ~50 values each, so a literal pickup+drop category would blow up to ~2,500
    # one-hot columns for a signal that ride_distance_km already captures more directly.
    # A point-in-time prior-trip count on the same route is a much lower-dimensional,
    # leakage-safe stand-in that still tells the model how well-travelled a route is.
    df["route_prior_bookings"] = df.groupby(["pickup_location", "drop_location"]).cumcount()

    return df.drop(columns=["_order"])


def customer_history_snapshot(customers):
    """Point-in-time customer features for a brand-new, not-yet-happened booking.

    Unlike training (where a booking's own outcome must be excluded from its features),
    scoring a hypothetical next booking can legitimately use a customer's full real
    history, since every booking behind it has already happened. customers.csv's raw
    totals are exactly that full history, so they are reused here (renamed and smoothed
    to match add_point_in_time_history's feature names) instead of being dropped.
    """
    global_cancel_rate = customers["cancellation_rate"].mean()
    snapshot = customers[["customer_id"]].copy()
    snapshot["customer_prior_bookings"] = customers["total_bookings"]
    snapshot["customer_prior_cancelled"] = customers["cancelled_rides"]
    snapshot["customer_prior_completed"] = customers["completed_rides"]
    snapshot["customer_prior_incomplete"] = customers["incomplete_rides"]
    snapshot["customer_prior_cancel_rate"] = _smoothed_rate(
        customers["cancelled_rides"], customers["total_bookings"], global_cancel_rate
    )
    return snapshot


def driver_history_snapshot(drivers):
    """Point-in-time driver features for a brand-new, not-yet-happened booking.

    See customer_history_snapshot for why reusing drivers.csv's full-history totals is
    valid at scoring time even though it is unsafe as a training feature.
    """
    global_incomplete_rate = (drivers["incomplete_rides"] / drivers["total_assigned_rides"]).mean()
    snapshot = drivers[["driver_id"]].copy()
    snapshot["driver_prior_bookings"] = drivers["total_assigned_rides"]
    snapshot["driver_prior_incomplete"] = drivers["incomplete_rides"]
    snapshot["driver_prior_incomplete_rate"] = _smoothed_rate(
        drivers["incomplete_rides"], drivers["total_assigned_rides"], global_incomplete_rate
    )
    return snapshot


def route_history_snapshot(bookings):
    """Point-in-time route-popularity feature for a brand-new, not-yet-happened booking.

    See customer_history_snapshot for why the full historical count from bookings.csv is
    valid to use here even though only a strictly-prior count is safe during training.
    """
    return (
        bookings.groupby(["pickup_location", "drop_location"])
        .size()
        .rename("route_prior_bookings")
        .reset_index()
    )


# Fixed-scale normalization caps for the composite scores below, chosen from the observed
# data ranges (driver_experience_years maxes at 14, customer_signup_days_ago at 999). Using
# fixed constants instead of a fitted scaler keeps the same formula usable at both training
# and live-scoring time without persisting extra state.
_EXPERIENCE_CAP_YEARS = 15
_SIGNUP_CAP_DAYS = 1000
_BOOKINGS_CAP = 20


def add_engineered_features(frame):
    """Add the project brief's remaining engineered features to an already-merged frame.

    Long_Distance_Flag is a direct threshold. Driver_Reliability_Score and
    Customer_Loyalty_Score are simple 0-1 weighted blends of the point-in-time history
    features plus static profile fields, so they carry no leakage beyond what their
    components already carry.
    """
    frame["long_distance_flag"] = (frame["ride_distance_km"] > 20).astype(int)

    frame["customer_loyalty_score"] = (
        (frame["customer_signup_days_ago"].clip(upper=_SIGNUP_CAP_DAYS) / _SIGNUP_CAP_DAYS) * 0.4
        + (frame["customer_prior_bookings"].clip(upper=_BOOKINGS_CAP) / _BOOKINGS_CAP) * 0.3
        + (frame["avg_customer_rating"] / 5) * 0.3
    )

    frame["driver_reliability_score"] = (
        (frame["driver_experience_years"].clip(upper=_EXPERIENCE_CAP_YEARS) / _EXPERIENCE_CAP_YEARS) * 0.25
        + (frame["avg_driver_rating"] / 5) * 0.25
        + frame["acceptance_rate"].clip(0, 1) * 0.25
        + (1 - frame["driver_prior_incomplete_rate"].clip(0, 1)) * 0.25
    )
    return frame


def prepare_frame(bookings, customers, drivers):
    frame = add_point_in_time_history(bookings)
    frame["booking_date"] = pd.to_datetime(frame["booking_date"])
    frame["booking_time"] = pd.to_datetime(frame["booking_time"], format="%H:%M:%S")
    frame["booking_hour"] = frame["booking_time"].dt.hour
    frame["booking_month"] = frame["booking_date"].dt.month
    frame["is_peak_time"] = frame["booking_hour"].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)
    frame["fare_per_km"] = frame["booking_value"] / frame["ride_distance_km"].clip(lower=0.01)
    frame["fare_markup"] = frame["booking_value"] / frame["base_fare"].clip(lower=0.01)
    frame["surge_cost"] = frame["surge_multiplier"] - 1
    customers_clean = customers.drop(columns=CUSTOMER_RAW_HISTORY_COLUMNS, errors="ignore")
    drivers_clean = drivers.drop(columns=DRIVER_RAW_LEAK_COLUMNS, errors="ignore")
    frame = frame.merge(customers_clean, on="customer_id", how="left", validate="many_to_one")
    frame = frame.merge(
        drivers_clean,
        on="driver_id",
        how="left",
        suffixes=("", "_driver"),
        validate="many_to_one",
    )
    return add_engineered_features(frame)


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

# booking_value ~= base_fare * surge_multiplier (R^2 = 0.997 with just that product), so
# base_fare is effectively the fare target in a thin disguise and must be excluded from the
# fare regression's inputs specifically (it is still a legitimate input for the other models).
FARE_ONLY_LEAKAGE_COLUMNS = ["base_fare"]

# driver_delay_flag is the driver-delay model's own target; it must not be used as a
# predictor in the other three models.
OTHER_TARGET_COLUMNS = ["driver_delay_flag"]


def tune_with_grid_search(estimator, param_grid, scoring, x_train, y_train, cv=3):
    """Run the project brief's required GridSearchCV step and return the best pipeline."""
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_


def classification_metrics(model_name, y_test, prediction, probabilities, labels):
    """Accuracy/precision/recall/F1 plus the AUC and confusion matrix the brief asks for.

    probabilities is the predict_proba output: shape (n, 2) for binary, (n, n_classes) for
    multiclass. labels fixes the class order so the confusion matrix is readable.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    metrics = {
        "model": model_name,
        "task": "Binary classification" if len(labels) == 2 else "Multiclass classification",
        "accuracy": accuracy_score(y_test, prediction),
        "precision_macro": precision_score(y_test, prediction, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, prediction, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, prediction, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=labels).tolist(),
        "confusion_matrix_labels": [str(label) for label in labels],
    }
    if len(labels) == 2:
        metrics["roc_auc"] = roc_auc_score(y_test, probabilities[:, 1])
        metrics["positive_class_recall"] = recall_score(
            y_test, prediction, pos_label=labels[1], zero_division=0
        )
    else:
        metrics["roc_auc"] = roc_auc_score(
            y_test, probabilities, multi_class="ovr", average="macro", labels=labels
        )
    return metrics
