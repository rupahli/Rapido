import json
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from model_common import (
    LEAKAGE_COLUMNS,
    MODEL_DIR,
    OTHER_TARGET_COLUMNS,
    classification_metrics,
    load_sources,
    make_preprocessor,
    prepare_frame,
    tune_with_grid_search,
)

MODEL_FILE = MODEL_DIR / "ride_outcome_model.pkl"
METRICS_FILE = MODEL_DIR / "ride_outcome_metrics.json"

PARAM_GRID = {
    "model__n_estimators": [150, 220],
    "model__max_depth": [12, 16],
    "model__min_samples_leaf": [4, 6],
}


def train_model():
    bookings, customers, drivers, _ = load_sources()
    frame = prepare_frame(bookings, customers, drivers)
    target = frame["booking_status"]
    features = frame.drop(columns=LEAKAGE_COLUMNS + OTHER_TARGET_COLUMNS, errors="ignore")
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    pipeline = Pipeline([
        ("preprocessor", make_preprocessor(features)),
        ("model", RandomForestClassifier(
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        )),
    ])
    model, best_params = tune_with_grid_search(
        pipeline, PARAM_GRID, scoring="f1_macro", x_train=x_train, y_train=y_train
    )
    prediction = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    labels = model.classes_.tolist()
    metrics = classification_metrics("Ride Outcome", y_test, prediction, probabilities, labels)
    metrics["best_params"] = best_params
    MODEL_DIR.mkdir(exist_ok=True)
    with MODEL_FILE.open("wb") as file:
        pickle.dump({"model": model, "columns": features.columns.tolist()}, file)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model, features.columns.tolist(), metrics


if __name__ == "__main__":
    print(train_model()[2])
