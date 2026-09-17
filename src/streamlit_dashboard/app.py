from pathlib import Path

import json
import numpy as np
import pandas as pd
import pickle
import streamlit as st

from model_common import (
    CUSTOMER_RAW_HISTORY_COLUMNS,
    DRIVER_RAW_LEAK_COLUMNS,
    MODEL_DIR,
    add_engineered_features,
    customer_history_snapshot,
    driver_history_snapshot,
    load_sources,
    prepare_frame,
    route_history_snapshot,
)
from train_cancellation_model import train_model as train_cancellation_model
from train_driver_delay_model import train_model as train_driver_delay_model
from train_fare_model import train_model as train_fare_model
from train_outcome_model import train_model as train_outcome_model


st.set_page_config(
    page_title="Rapido Intelligent Mobility Insights",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).resolve().parents[1] / "DataCSVFiles"
MODEL_DIR = Path(__file__).resolve().parent / "model_artifacts"
MODEL_FILE = MODEL_DIR / "rapido_models.joblib"
MODEL_VERSION = "2026-09-10-v1"


def load_data():
    return load_sources()


def booking_features(bookings):
    frame = bookings.copy()
    frame["booking_date"] = pd.to_datetime(frame["booking_date"])
    frame["booking_time"] = pd.to_datetime(frame["booking_time"], format="%H:%M:%S")
    frame["booking_hour"] = frame["booking_time"].dt.hour
    frame["booking_month"] = frame["booking_date"].dt.month
    frame["is_peak_time"] = frame["booking_hour"].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)
    frame["fare_per_km"] = frame["booking_value"] / frame["ride_distance_km"].clip(lower=0.01)
    frame["fare_markup"] = frame["booking_value"] / frame["base_fare"].clip(lower=0.01)
    frame["surge_cost"] = frame["surge_multiplier"] - 1
    return frame


def attach_customer_driver_history(frame, customers, drivers, all_bookings):
    """Merge customer/driver/route context onto a hypothetical new booking row.

    Mirrors model_common.prepare_frame's column set (raw history columns dropped, replaced
    by the *_prior_* snapshot columns, plus the engineered composite scores) so the row
    lines up with what each model was trained on. Reusing customers.csv/drivers.csv/
    bookings.csv's full history here is safe: for a booking that has not happened yet,
    every booking behind it is real past data (see model_common.customer_history_snapshot
    for the training-vs-scoring distinction). all_bookings is the full bookings.csv table,
    used only to look up how well-travelled the row's route is.
    """
    customers_clean = customers.drop(columns=CUSTOMER_RAW_HISTORY_COLUMNS, errors="ignore")
    drivers_clean = drivers.drop(columns=DRIVER_RAW_LEAK_COLUMNS, errors="ignore")
    frame = frame.merge(customers_clean, on="customer_id", how="left")
    frame = frame.merge(customer_history_snapshot(customers), on="customer_id", how="left")
    frame = frame.merge(drivers_clean, on="driver_id", how="left", suffixes=("", "_driver"))
    frame = frame.merge(driver_history_snapshot(drivers), on="driver_id", how="left")
    frame = frame.merge(
        route_history_snapshot(all_bookings), on=["pickup_location", "drop_location"], how="left"
    )
    frame["route_prior_bookings"] = frame["route_prior_bookings"].fillna(0)
    return add_engineered_features(frame)


def train_models():
    bookings, customers, drivers, locations = load_data()
    model_specs = {
        "cancellation": ("customer_cancellation_model.pkl", "customer_cancellation_metrics.json", train_cancellation_model),
        "outcome": ("ride_outcome_model.pkl", "ride_outcome_metrics.json", train_outcome_model),
        "fare": ("fare_model.pkl", "fare_metrics.json", train_fare_model),
        "delay": ("driver_delay_model.pkl", "driver_delay_metrics.json", train_driver_delay_model),
    }
    bundle = {"frame": prepare_frame(bookings, customers, drivers), "locations": locations, "metrics": {}}
    for name, (model_name, metrics_name, train_model) in model_specs.items():
        model_path = MODEL_DIR / model_name
        metrics_path = MODEL_DIR / metrics_name
        if not model_path.exists() or not metrics_path.exists():
            train_model()
        with model_path.open("rb") as file:
            saved_model = pickle.load(file)
        bundle[name] = (saved_model["model"], saved_model["columns"])
        bundle["metrics"][name] = json.loads(metrics_path.read_text(encoding="utf-8"))
    return bundle


def score_row(model_bundle, values):
    model, columns = model_bundle
    row = pd.DataFrame([values]).reindex(columns=columns)
    return model, row


st.title("Rapido Intelligent Mobility Insights")
#st.caption("Decision support for ride outcomes, fares, customer cancellation risk, and driver reliability")

try:
    bookings, customers, drivers, locations = load_data()
    bundle = train_models()
except Exception as error:
    st.error(f"Dashboard could not load the Rapido data: {error}")
    st.stop()

with st.sidebar:
    st.subheader("KPIs")
    st.metric("Bookings", f"{len(bookings):,}")
    st.metric("Cancellation rate", f"{(bookings['booking_status'].eq('Cancelled').mean() * 100):.1f}%")
    st.metric("Cities", bookings["city"].nunique())
    st.divider()
    #st.caption("Models are loaded from disk after the first training run.")

tab_overview, tab_performance, tab_cancellation, tab_outcome, tab_fare, tab_driver = st.tabs(
    ["Overview", "Model performance", "Customer risk", "Ride outcomes", "Fare forecast", "Driver reliability"]
)

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

with tab_overview:
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Ride patterns by hour")
        hourly = bookings.groupby("hour_of_day").size().rename("Bookings")
        st.line_chart(hourly)
    with right:
        st.subheader("Outcome mix")
        outcome_mix = bookings["booking_status"].value_counts().rename_axis("Status").to_frame("Bookings")
        st.bar_chart(outcome_mix)

    left, right = st.columns(2)
    with left:
        st.subheader("Ride volume by weekday")
        weekday_volume = bookings.groupby("day_of_week").size().reindex(WEEKDAY_ORDER).rename("Bookings")
        st.bar_chart(weekday_volume)
    with right:
        st.subheader("Cancellation by city")
        city_cancel = bookings.assign(cancelled=bookings["booking_status"].eq("Cancelled")).groupby("city")["cancelled"].mean().sort_values(ascending=False)
        st.bar_chart(city_cancel)

    
    left, right = st.columns(2)
    with left:
        st.subheader("Distance versus fare")
        distance_fare = bookings[["ride_distance_km", "booking_value"]].copy()
       
        st.scatter_chart(distance_fare, x="ride_distance_km", y="booking_value")
        st.caption(f"Correlation: {bookings['ride_distance_km'].corr(bookings['booking_value']):.2f}")
    with right:
        st.subheader("Rating distributions")
        rating_counts = pd.DataFrame(
            {
                "Customer rating": customers["avg_customer_rating"].value_counts(),
                "Driver rating": drivers["avg_driver_rating"].value_counts(),
            }
        ).fillna(0).sort_index()
        rating_counts.index.name = "Rating"
        st.line_chart(rating_counts)

    left, right = st.columns(2)
    with left:
        st.subheader("Customer versus driver behavior")
        behavior = pd.DataFrame(
            {
                "Customer cancellation rate": [customers["cancellation_rate"].mean()],
                "Driver delay rate": [drivers["delay_rate"].mean()],
                "Driver acceptance rate": [drivers["acceptance_rate"].mean()],
            }
        ).T.rename(columns={0: "Rate"})
        st.bar_chart(behavior)
    with right:
        st.subheader("Surge behavior")
        surge_pattern = bookings.groupby("hour_of_day")["surge_multiplier"].mean()
        st.line_chart(surge_pattern)

    st.subheader("Pickup and drop activity")
    location_activity = bookings.groupby(["pickup_location", "drop_location"]).size().nlargest(20).rename("Bookings")
    st.dataframe(location_activity.reset_index(), use_container_width=True, hide_index=True)
    #if "payment_method" not in bookings.columns:
      #  st.caption("Payment-method usage is unavailable because the source bookings file has no payment_method column.")

with tab_performance:
    st.subheader("Saved model performance")
    #st.caption("Metrics are calculated on the held-out test split during each model's training script.")
    metrics = bundle["metrics"]
    performance_rows = []
    for name, values in metrics.items():
        row = {"Model": values["model"], "Task": values["task"]}
        if values["task"] == "Regression":
            row.update({"RMSE": values["rmse"], "MAE": values["mae"], "R2": values["r2"]})
        else:
            row.update({
                "Accuracy": values["accuracy"],
                "Precision (macro)": values["precision_macro"],
                "Recall (macro)": values["recall_macro"],
                "F1 (macro)": values["f1_macro"]#,
                #"AUC": values.get("roc_auc"),
            })
        performance_rows.append(row)
    performance_table = pd.DataFrame(performance_rows).set_index("Model")
    st.dataframe(
        performance_table.style.format({
            "Accuracy": "{:.1%}",
            "Precision (macro)": "{:.1%}",
            "Recall (macro)": "{:.1%}",
            "F1 (macro)": "{:.1%}",
            #"AUC": "{:.3f}",
            "RMSE": "{:.2f}",
            "MAE": "{:.2f}",
            "R2": "{:.3f}",
        }),
        use_container_width=True,
    )
    #st.subheader("Classification comparison")
    #classification_metrics = performance_table.drop(columns=["RMSE", "MAE", "R2", "AUC"], errors="ignore")
    #st.bar_chart(classification_metrics)
    #st.info("Accuracy, precision, recall, and F1 are macro-averaged for classification models so minority classes are represented fairly. AUC is macro one-vs-rest for the 3-class outcome model. Fare forecasting uses RMSE, MAE, and R2.")

    st.subheader("Confusion matrices")
    #st.caption("Rows are the actual outcome, columns are the model's prediction, on the held-out test split.")
    confusion_cols = st.columns(3)
    classification_models = [
        (name, values) for name, values in metrics.items() if "confusion_matrix" in values
    ]
    for col, (name, values) in zip(confusion_cols, classification_models):
        with col:
            st.markdown(f"**{values['model']}**")
            labels = values["confusion_matrix_labels"]
            matrix = pd.DataFrame(
                values["confusion_matrix"],
                index=[f"Actual {label}" for label in labels],
                columns=[f"Predicted {label}" for label in labels],
            )
            st.dataframe(matrix, use_container_width=True)

with tab_cancellation:
    st.subheader("Customer cancellation risk")
    #st.write("Estimate cancellation probability from customer history, booking conditions, peak-time behavior, and pricing signals.")
    customer_id = st.selectbox("Customer", sorted(customers["customer_id"].unique()))
    customer = customers.loc[customers["customer_id"].eq(customer_id)].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Historical cancellation rate", f"{customer['cancellation_rate']:.1%}")
    c2.metric("Past bookings", int(customer["total_bookings"]))
    c3.metric("Average rating", f"{customer['avg_customer_rating']:.1f}")
    booking = bookings.iloc[0].copy()
    booking["customer_id"] = customer_id
    row = attach_customer_driver_history(booking_features(pd.DataFrame([booking])), customers, drivers, bookings)
    values = row.iloc[0].to_dict()
    model, columns = bundle["cancellation"]
    probability = float(model.predict_proba(pd.DataFrame([values]).reindex(columns=columns))[0, 1])
    st.metric("Predicted cancellation probability", f"{probability:.1%}")
    #st.progress(min(probability, 1.0))
    #st.info("Use the probability as a prioritization signal, not as a guarantee of customer intent.")

with tab_outcome:
    st.subheader("Ride outcome prediction")
    #st.write("Estimate whether a booking is likely to complete, cancel, or become incomplete.")
    row = attach_customer_driver_history(booking_features(bookings.iloc[[0]]), customers, drivers, bookings)
    model, columns = bundle["outcome"]
    probabilities = model.predict_proba(row.reindex(columns=columns))[0]
    outcome_table = pd.DataFrame({"Outcome": model.classes_, "Probability": probabilities}).set_index("Outcome")
    st.bar_chart(outcome_table)
    st.dataframe(outcome_table.style.format({"Probability": "{:.1%}"}), use_container_width=True)

with tab_fare:
    st.subheader("Fare forecast")
    distance = st.number_input("Ride distance (km)", min_value=0.1, value=8.0, step=0.5)
    surge = st.slider("Surge multiplier", 1.0, 3.0, 1.4, 0.1)
    vehicle = st.selectbox("Vehicle type", sorted(bookings["vehicle_type"].unique()))
    fare_row = bookings.iloc[[0]].copy()
    fare_row["ride_distance_km"] = distance
    fare_row["surge_multiplier"] = surge
    fare_row["vehicle_type"] = vehicle
    fare_row = attach_customer_driver_history(booking_features(fare_row), customers, drivers, bookings)
    model, columns = bundle["fare"]
    estimate = float(model.predict(fare_row.reindex(columns=columns))[0])
    st.metric("Estimated booking value", f"Rs {estimate:,.2f}")
    #st.caption("This forecast is based on the trained model")

with tab_driver:
    st.subheader("Driver reliability")
    driver_id = st.selectbox("Driver", sorted(drivers["driver_id"].unique()))
    driver = drivers.loc[drivers["driver_id"].eq(driver_id)].iloc[0]
    d1, d2, d3 = st.columns(3)
    d1.metric("Acceptance rate", f"{driver['acceptance_rate']:.1%}")
    d2.metric("Delay rate", f"{driver['delay_rate']:.1%}")
    d3.metric("Driver rating", f"{driver['avg_driver_rating']:.1f}")
    row = bookings.iloc[[0]].copy()
    row["driver_id"] = driver_id
    row = attach_customer_driver_history(booking_features(row), customers, drivers, bookings)
    model, columns = bundle["delay"]
    delay_probability = float(model.predict_proba(row.reindex(columns=columns))[0, 1])
    st.metric("Predicted delay/incomplete risk", f"{delay_probability:.1%}")
    #st.progress(min(delay_probability, 1.0))
