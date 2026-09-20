from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


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

PLOT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "plots"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(DATA_FILE)

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


# ---------------------------------------------------------
# ENCODE TARGET
# ---------------------------------------------------------

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)


# ---------------------------------------------------------
# SAME TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded
)


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

model_files = {

    "Decision Tree":
        MODEL_DIR / "decision_tree_model.pkl",

    "Random Forest":
        MODEL_DIR / "random_forest_model.pkl",

    "XGBoost":
        MODEL_DIR / "xgboost_model.pkl"
}


results = []


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

for name, model_file in model_files.items():

    print("\n" + "=" * 70)

    print(name)

    print("=" * 70)


    model = joblib.load(model_file)


    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=encoder.classes_,
            zero_division=0
        )
    )


    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )


    print("Confusion Matrix:")

    print(cm)


    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=encoder.classes_
    )


    display.plot()

    plt.title(
        f"{name} - Confusion Matrix"
    )

    plt.tight_layout()


    safe_name = (
        name.lower()
        .replace(" ", "_")
    )


    plt.savefig(
        PLOT_DIR
        / f"{safe_name}_confusion_matrix.png",
        dpi=300
    )


    plt.close()


    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1

    })


# ---------------------------------------------------------
# RESULTS TABLE
# ---------------------------------------------------------

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 70)

print("MODEL COMPARISON")

print("=" * 70)

print(results_df)


results_df.to_csv(
    RESULT_DIR
    / "model_comparison.csv",
    index=False
)


print(
    "\nResults saved to:"
)

print(
    RESULT_DIR
    / "model_comparison.csv"
)
# ---------------------------------------------------------
# STEP 6 - SELECT BEST MODEL
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 6 - SELECTING BEST MODEL")
print("=" * 70)


# Find model with highest F1 Score

best_index = results_df["F1_Score"].idxmax()

best_model_name = results_df.loc[
    best_index,
    "Model"
]

best_f1 = results_df.loc[
    best_index,
    "F1_Score"
]


print("\nSelected Model:")
print(best_model_name)

print(
    f"F1 Score: {best_f1:.4f}"
)


# ---------------------------------------------------------
# MODEL FILE MAPPING
# ---------------------------------------------------------

model_file_mapping = {

    "Decision Tree":
        MODEL_DIR / "decision_tree_model.pkl",

    "Random Forest":
        MODEL_DIR / "random_forest_model.pkl",

    "XGBoost":
        MODEL_DIR / "xgboost_model.pkl"
}


best_model_file = model_file_mapping[
    best_model_name
]


# ---------------------------------------------------------
# LOAD BEST MODEL
# ---------------------------------------------------------

best_model = joblib.load(
    best_model_file
)


# ---------------------------------------------------------
# SAVE FINAL MODEL
# ---------------------------------------------------------

final_model_file = (
    MODEL_DIR
    / "congestion_model.pkl"
)


joblib.dump(
    best_model,
    final_model_file
)


# ---------------------------------------------------------
# SAVE MODEL INFORMATION
# ---------------------------------------------------------

model_info = {

    "model_name":
        best_model_name,

    "f1_score":
        float(best_f1),

    "model_file":
        str(best_model_file)
}


joblib.dump(
    model_info,
    MODEL_DIR / "model_info.pkl"
)


print("\nFinal model saved:")
print(final_model_file)

print("\nSTEP 6 COMPLETED")