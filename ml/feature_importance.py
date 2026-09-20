from pathlib import Path

import pandas as pd
import joblib
import matplotlib.pyplot as plt


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

PLOT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "plots"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "model_results"
)

PLOT_DIR.mkdir(
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

df = pd.read_csv(DATA_FILE)


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


# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

all_results = []


for name, model_file in model_files.items():

    model = joblib.load(
        model_file
    )


    importance = model.feature_importances_


    importance_df = pd.DataFrame({

        "Feature":
            feature_columns,

        "Importance":
            importance

    })


    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
    )


    print("\n" + "=" * 70)

    print(name)

    print("=" * 70)

    print(importance_df)


    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    safe_name = (
        name.lower()
        .replace(" ", "_")
    )


    importance_df.to_csv(

        RESULT_DIR
        / f"{safe_name}_feature_importance.csv",

        index=False
    )


    # -----------------------------------------------------
    # GRAPH
    # -----------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )


    plt.barh(

        importance_df["Feature"],

        importance_df["Importance"]

    )


    plt.gca().invert_yaxis()


    plt.title(
        f"{name} - Feature Importance"
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Network Feature"
    )


    plt.tight_layout()


    plt.savefig(

        PLOT_DIR
        / f"{safe_name}_feature_importance.png",

        dpi=300
    )


    plt.close()


print("\n" + "=" * 70)

print("FEATURE IMPORTANCE ANALYSIS COMPLETED")

print("=" * 70)