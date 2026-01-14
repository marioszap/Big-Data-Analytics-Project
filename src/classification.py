from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # bool -> int
    for c in X.select_dtypes(include="bool").columns:
        X[c] = X[c].astype(int)

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def run_classification() -> None:
    clinical_path = OUT_DIR / "clinical_clean.csv"
    if not clinical_path.exists():
        raise FileNotFoundError("Run preprocessing first to generate clinical_clean.csv")

    df = pd.read_csv(clinical_path, sep=";")

    target_col = "fried"
    forbidden = [
        "weight_loss",
        "exhaustion_score",
        "gait_speed_slower",
        "grip_strength_abnormal",
        "low_physical_activity",
    ]

    drop_cols = ["part_id", target_col] + forbidden
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols).copy()
    y = df[target_col].copy()

    # Train/test split (final evaluation) + CV on train only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train.copy())

    # Models + grids
    models = {
        "logreg": (
            LogisticRegression(max_iter=4000, class_weight="balanced"),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs"],
            }
        ),
        "rf": (
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2"],
            }
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, (estimator, grid) in models.items():
        pipe = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", estimator),
        ])

        gs = GridSearchCV(
            pipe,
            param_grid=grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True
        )
        gs.fit(X_train, y_train)

        best_model = gs.best_estimator_
        y_pred = best_model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        f1m = float(f1_score(y_test, y_pred, average="macro"))
        cm = confusion_matrix(y_test, y_pred).tolist()

        print("\n==============================")
        print(f"Model: {name}")
        print("Best params:", gs.best_params_)
        print(f"CV best macro-F1: {gs.best_score_:.4f}")
        print(f"Test accuracy: {acc:.4f}")
        print(f"Test macro-F1: {f1m:.4f}")
        print("Confusion matrix:\n", np.array(cm))
        print("\nClassification report:\n", classification_report(y_test, y_pred))

        results[name] = {
            "best_params": gs.best_params_,
            "cv_best_macro_f1": float(gs.best_score_),
            "test_accuracy": acc,
            "test_macro_f1": f1m,
            "confusion_matrix": cm,
            "dropped_columns": drop_cols,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        }

    save_json(results, OUT_DIR / "classification_results.json")
    print(f"\n✅ Saved: {OUT_DIR / 'classification_results.json'}")


if __name__ == "__main__":
    run_classification()
