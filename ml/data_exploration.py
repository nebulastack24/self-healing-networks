from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "network_metrics.csv"


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

print("=" * 70)
print("SELF-HEALING CAMPUS NETWORKS")
print("STEP 1 - DATASET EXPLORATION")
print("=" * 70)

if not DATA_FILE.exists():
    print(f"\nERROR: Dataset not found:")
    print(DATA_FILE)
    raise SystemExit

df = pd.read_csv(DATA_FILE)


# ---------------------------------------------------------
# BASIC INFORMATION
# ---------------------------------------------------------

print("\n1. DATASET SHAPE")
print("-" * 40)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


print("\n2. COLUMN NAMES")
print("-" * 40)

for column in df.columns:
    print(column)


print("\n3. FIRST 5 ROWS")
print("-" * 40)
print(df.head())


print("\n4. DATA TYPES")
print("-" * 40)
print(df.dtypes)


print("\n5. MISSING VALUES")
print("-" * 40)

missing = df.isnull().sum()

print(missing)


print("\n6. DUPLICATE ROWS")
print("-" * 40)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates}")


print("\n7. NUMERICAL SUMMARY")
print("-" * 40)

print(df.describe())


print("\n8. UNIQUE VALUES")
print("-" * 40)

for column in df.columns:
    if df[column].dtype == "object":
        print(f"\n{column}:")
        print(df[column].unique())


print("\n9. DATASET INFORMATION")
print("-" * 40)

df.info()


print("\n" + "=" * 70)
print("STEP 1 COMPLETED")
print("=" * 70)