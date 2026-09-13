import json
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from model_common import LEAKAGE_COLUMNS, MODEL_DIR, load_sources, make_preprocessor, prepare_frame

MODEL_FILE = MODEL_DIR / "fare_model.pkl"
METRICS_FILE = MODEL_DIR / "fare_metrics.json"


def train_model():
    bookings, customers, drivers, _ = load_sources()
    frame = prepare_frame(bookings, customers, drivers)
    target = frame["booking_value"]
    features = frame.drop(
        columns=LEAKAGE_COLUMNS + ["booking_value", "fare_per_km", "fare_markup"],
        errors="ignore",
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = Pipeline([
        ("preprocessor", make_preprocessor(features)),
        ("model", RandomForestRegressor(
            n_estimators=150, max_depth=14, min_samples_leaf=4,
            random_state=42, n_jobs=-1
        )),
    ])
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    metrics = {
        "model": "Fare Forecast",
        "task": "Regression",
        "rmse": mean_squared_error(y_test, prediction) ** 0.5,
        "mae": mean_absolute_error(y_test, prediction),
        "r2": r2_score(y_test, prediction),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    with MODEL_FILE.open("wb") as file:
        pickle.dump({"model": model, "columns": features.columns.tolist()}, file)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model, features.columns.tolist(), metrics


if __name__ == "__main__":
    print(train_model()[2])
