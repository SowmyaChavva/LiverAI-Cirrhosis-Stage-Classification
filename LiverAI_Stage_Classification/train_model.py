from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "liver_cirrhosis.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "liver_stage_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"

NUMERIC_FIELDS = [
    "N_Days", "Age", "Bilirubin", "Cholesterol", "Albumin", "Copper",
    "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"
]
CATEGORICAL_FIELDS = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]
FEATURES = NUMERIC_FIELDS + CATEGORICAL_FIELDS
TARGET = "Stage"


def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit("Missing required columns: " + ", ".join(missing))

    original_rows = len(df)
    df = df[FEATURES + [TARGET]].copy()  # Exclude Status to reduce outcome leakage risk.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([1, 2, 3])]
    df[TARGET] = df[TARGET].astype(int)

    before_dedup = len(df)
    df = df.drop_duplicates()
    removed_duplicates = before_dedup - len(df)
    if df.empty:
        raise SystemExit("No usable stage-labelled rows found.")
    counts = df[TARGET].value_counts()
    if set(counts.index.tolist()) != {1, 2, 3} or counts.min() < 2:
        raise SystemExit(f"Need at least two unique examples of each stage. Counts: {counts.to_dict()}")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FIELDS),
        ("categorical", categorical_pipe, CATEGORICAL_FIELDS)
    ])
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "original_rows": int(original_rows),
        "rows_after_label_filter": int(before_dedup),
        "duplicates_removed": int(removed_duplicates),
        "unique_rows_used": int(len(df)),
        "class_counts_after_dedup": {str(k): int(v) for k, v in counts.sort_index().items()},
        "classes": [1, 2, 3],
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
        "confusion_matrix_labels_1_2_3": confusion_matrix(y_test, y_pred, labels=[1, 2, 3]).tolist(),
        "classification_report": report,
        "caveat": "Educational evaluation only. Dataset provenance and clinical validity must be verified; not for diagnosis."
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Original rows: {original_rows}")
    print(f"Duplicates removed: {removed_duplicates}")
    print(f"Unique rows used: {len(df)}")
    print(f"Stage counts: {counts.sort_index().to_dict()}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Saved model:", MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)


if __name__ == "__main__":
    main()
