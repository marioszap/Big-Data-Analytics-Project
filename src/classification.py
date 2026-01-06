from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# Paths (robust)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Build a fresh preprocessor (avoid shared state across models)."""
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if len(num_cols) > 0:
        transformers.append(("num", numeric_pipe, num_cols))
    if len(cat_cols) > 0:
        transformers.append(("cat", categorical_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


if __name__ == "__main__":

    # -----------------------------
    # Load data
    # -----------------------------
    clinical_path = OUT_DIR / "clinical_clean.csv"
    if not clinical_path.exists():
        raise FileNotFoundError(
            f"Could not find {clinical_path}. Run preprocessing first to generate clinical_clean.csv."
        )

    df = pd.read_csv(clinical_path, sep=";")

    # -----------------------------
    # Target + forbidden features
    # -----------------------------
    target_col = "fried"
    forbidden = [
        "weight_loss",
        "exhaustion_score",
        "gait_speed_slower",
        "grip_strength_abnormal",
        "low_physical_activity",
    ]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns.")

    # -----------------------------
    # Build X / y
    # -----------------------------
    #drop_cols = ["part_id"] + forbidden
    drop_cols = ["part_id", target_col] + forbidden
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # FIX: bool -> int (SimpleImputer doesn't support bool)
    bool_cols = X.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    # -----------------------------
    # Train/Test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # Identify numeric vs categorical
    # -----------------------------
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # -----------------------------
    # Models
    # -----------------------------
    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "rf": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced",
        ),
    }

    results: dict = {}

    for name, model in models.items():
        preprocessor = build_preprocessor(num_cols, cat_cols)

        clf = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", model),
        ])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        f1_macro = float(f1_score(y_test, y_pred, average="macro"))
        cm = confusion_matrix(y_test, y_pred).tolist()

        print("\n==============================")
        print(f"Model: {name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro-F1: {f1_macro:.4f}")
        print("Confusion matrix:")
        print(np.array(cm))
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))

        results[name] = {
            "accuracy": acc,
            "macro_f1": f1_macro,
            "confusion_matrix": cm,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "num_features": len(num_cols),
            "cat_features": len(cat_cols),
            "dropped_columns": drop_cols,
        }

        # ------------------------------------------
        # Feature importance (Random Forest only)
        # ------------------------------------------
        if name == "rf":
            importances = clf.named_steps["model"].feature_importances_

            # Robust feature names:
            # If cat_cols is empty, only numeric features exist.
            if len(cat_cols) == 0:
                feature_names = np.array(num_cols, dtype=object)
            else:
                # Works in modern sklearn: returns names like num__age, cat__gender_F, ...
                feature_names = clf.named_steps["prep"].get_feature_names_out()

            if len(feature_names) != len(importances):
                raise RuntimeError(
                    f"Feature names ({len(feature_names)}) and importances ({len(importances)}) length mismatch."
                )

            feat_imp_df = (
                pd.DataFrame({"feature": feature_names, "importance": importances})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )

            feat_imp_df.to_csv(OUT_DIR / "feature_importance_rf.csv", index=False)

            print("\nTop 15 most important features (RF):")
            print(feat_imp_df.head(15))

    save_json(results, OUT_DIR / "classification_results.json")
    print(f"\n✅ Saved results to {OUT_DIR / 'classification_results.json'}")
    if (OUT_DIR / "feature_importance_rf.csv").exists():
        print(f"✅ Saved RF feature importance to {OUT_DIR / 'feature_importance_rf.csv'}")
