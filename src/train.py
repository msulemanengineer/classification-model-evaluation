
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# Load the public dataset.
dataset = load_breast_cancer(as_frame=True)
X = dataset.data

# In scikit-learn, 0 means malignant and 1 means benign.
# Define malignant as the positive class (1) for this experiment.
y = (dataset.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

models = {
    "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000)),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
}

all_metrics = {}
error_frames = []

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Probability of the positive class: malignant.
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }

    all_metrics[name] = metrics

    errors = X_test.loc[y_test != predictions].copy()
    errors.insert(0, "model", name)
    errors["actual_malignant"] = y_test.loc[y_test != predictions]
    errors["predicted_malignant"] = predictions[
        y_test.to_numpy() != predictions
    ]
    error_frames.append(errors.reset_index(names="sample_id"))

    print(f"\n{name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

with open(RESULTS_DIR / "metrics.json", "w") as file:
    json.dump(all_metrics, file, indent=2)

pd.concat(error_frames, ignore_index=True).to_csv(
    RESULTS_DIR / "error_analysis.csv",
    index=False,
)

print("\nResults saved in the results/ directory.")