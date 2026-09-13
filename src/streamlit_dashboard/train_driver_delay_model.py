import json
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from model_common import LEAKAGE_COLUMNS, MODEL_DIR, load_sources, make_preprocessor, prepare_frame

MODEL_FILE = MODEL_DIR / "driver_delay_model.pkl"
METRICS_FILE = MODEL_DIR / "driver_delay_metrics.json"


def train_model():
    bookings, customers, drivers, _ = load_sources()
    frame = prepare_frame(bookings, customers, drivers)
    target = frame["driver_delay_flag"]
    features = frame.drop(columns=["driver_delay_flag"] + LEAKAGE_COLUMNS, errors="ignore")
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    model = Pipeline([
        ("preprocessor", make_preprocessor(features)),
        ("model", RandomForestClassifier(
            n_estimators=180, max_depth=14, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        )),
    ])
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    metrics = {
        "model": "Driver Delay Risk",
        "task": "Binary classification",
        "accuracy": accuracy_score(y_test, prediction),
        "precision_macro": precision_score(y_test, prediction, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, prediction, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, prediction, average="macro", zero_division=0),
        "positive_class_recall": recall_score(y_test, prediction, zero_division=0),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    with MODEL_FILE.open("wb") as file:
        pickle.dump({"model": model, "columns": features.columns.tolist()}, file)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model, features.columns.tolist(), metrics


if __name__ == "__main__":
    print(train_model()[2])
