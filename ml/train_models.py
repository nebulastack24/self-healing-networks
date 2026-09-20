from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "ml_dataset.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "model_results"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("=" * 70)
print("STEP 5 - MACHINE LEARNING MODEL TRAINING")
print("=" * 70)

if not DATA_FILE.exists():

    print("\nERROR:")
    print("ml_dataset.csv was not found.")

    print(
        "\nRun this first:"
    )

    print(
        "python ml\\data_preprocessing.py"
    )

    raise SystemExit


df = pd.read_csv(DATA_FILE)


# ---------------------------------------------------------
# TARGET CHECK
# ---------------------------------------------------------

if "congestion_level" not in df.columns:

    print(
        "\nERROR: congestion_level column "
        "does not exist."
    )

    raise SystemExit


# ---------------------------------------------------------
# REMOVE MISSING TARGET
# ---------------------------------------------------------

df = df.dropna(
    subset=["congestion_level"]
)


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

feature_columns = [
    "connected_devices",
    "upload_mbps",
    "download_mbps",
    "latency_ms",
    "packet_loss_percent",
    "bandwidth_usage_percent",
    "throughput_mbps",
    "hour",
    "minute",
    "day_of_week"
]


feature_columns = [
    column
    for column in feature_columns
    if column in df.columns
]


X = df[feature_columns]

y = df["congestion_level"]


print("\nFeatures:")
for feature in feature_columns:
    print(f"  {feature}")


print("\nTarget distribution:")
print(y.value_counts())


# ---------------------------------------------------------
# ENCODE TARGET
# ---------------------------------------------------------

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


print("\nClass mapping:")

for index, label in enumerate(
    label_encoder.classes_
):
    print(
        f"{label} -> {index}"
    )


# ---------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

models = {

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42,
            max_depth=8
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=12,
            n_jobs=-1
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            eval_metric="mlogloss"
        )
}


# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------

trained_models = {}


for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )

    model.fit(
        X_train,
        y_train
    )

    trained_models[name] = model

    safe_name = (
        name.lower()
        .replace(" ", "_")
    )

    model_file = (
        MODEL_DIR
        / f"{safe_name}_model.pkl"
    )

    encoder_file = (
        MODEL_DIR
        / "label_encoder.pkl"
    )

    joblib.dump(
        model,
        model_file
    )

    joblib.dump(
        label_encoder,
        encoder_file
    )

    print(
        f"Saved: {model_file}"
    )


# ---------------------------------------------------------
# SAVE TEST DATA
# ---------------------------------------------------------

test_df = X_test.copy()

test_df["actual_encoded"] = y_test

test_df.to_csv(
    RESULT_DIR / "test_data.csv",
    index=False
)


print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED")
print("=" * 70)