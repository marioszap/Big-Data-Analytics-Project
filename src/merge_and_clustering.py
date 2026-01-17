# MERGED DATASET ONLY
# Outputs (like your screenshots):
# - Best KMeans run across PCA choices (best k, silhouette)
# - Best AGNES (Agglomerative Ward) run across PCA choices
# - Best GMM run across PCA choices (also tries covariance_type=full/diag)
#
# Also saves:
#   data/outputs/merged_dataset.csv
#   data/outputs/clustering_runs_merged_kmeans_agnes_gmm.csv
#   data/outputs/clustered_merged_best_kmeans.csv
#   data/outputs/clustered_merged_best_agnes.csv
#   data/outputs/clustered_merged_best_gmm.csv
#   data/outputs/cluster_fried_percent_best_*.csv  (interpretation only)
#   data/outputs/clustering_report_merged_kmeans_agnes_gmm.json
# ======================================================

from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def merge_datasets() -> pd.DataFrame:
    beacons_path = OUT_DIR / "beacons_features.csv"
    clinical_path = OUT_DIR / "clinical_clean.csv"

    if not beacons_path.exists():
        raise FileNotFoundError(f"Missing {beacons_path}. Run preprocessing first.")
    if not clinical_path.exists():
        raise FileNotFoundError(f"Missing {clinical_path}. Run preprocessing first.")

    beacons = pd.read_csv(beacons_path, sep=";")
    clinical = pd.read_csv(clinical_path, sep=";")

    beacons["part_id"] = beacons["part_id"].astype(str).str.strip()
    clinical["part_id"] = clinical["part_id"].astype(str).str.strip()

    merged = beacons.merge(clinical, on="part_id", how="inner")
    merged.to_csv(OUT_DIR / "merged_dataset.csv", sep=";", index=False)
    return merged


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

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


def make_X_merged_only(merged: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:

    forbidden = [
        "weight_loss",
        "exhaustion_score",
        "gait_speed_slower",
        "grip_strength_abnormal",
        "low_physical_activity",
    ]

    drop_cols = ["part_id"]
    if "fried" in merged.columns:
        drop_cols.append("fried")
    drop_cols += [c for c in forbidden if c in merged.columns]

    X = merged.drop(columns=[c for c in drop_cols if c in merged.columns]).copy()
    return X, X.columns.tolist(), drop_cols


def evaluate_grid_kmeans_agnes_gmm(
    X_dense: np.ndarray,
    pca_list: List[Optional[int]],
    k_list: List[int],
    gmm_cov_types: List[str] = ["full", "diag"],
) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:

    all_runs = []
    best_kmeans = None
    best_agnes = None
    best_gmm = None

    for pca_k in pca_list:
        if pca_k is None:
            X_use = X_dense
            pca_info = {"pca": None, "explained_var": None}
        else:
            pca_k_eff = int(min(pca_k, X_dense.shape[1]))
            pca = PCA(n_components=pca_k_eff, random_state=42)
            X_use = pca.fit_transform(X_dense)
            pca_info = {"pca": pca_k_eff, "explained_var": float(np.sum(pca.explained_variance_ratio_))}

        for k in k_list:
            model = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = model.fit_predict(X_use)
            s = float(silhouette_score(X_use, labels)) if len(np.unique(labels)) > 1 else float("nan")

            run = {"method": "KMeans", "k": k, "silhouette": s, **pca_info}
            all_runs.append(run)

            if not np.isnan(s) and (best_kmeans is None or s > best_kmeans["silhouette"]):
                best_kmeans = {**run, "labels": labels}

        for k in k_list:
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels = model.fit_predict(X_use)
            s = float(silhouette_score(X_use, labels)) if len(np.unique(labels)) > 1 else float("nan")

            run = {"method": "AGNES", "k": k, "silhouette": s, **pca_info}
            all_runs.append(run)

            if not np.isnan(s) and (best_agnes is None or s > best_agnes["silhouette"]):
                best_agnes = {**run, "labels": labels}

        for cov in gmm_cov_types:
            for k in k_list:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=cov,
                    random_state=42
                )
                labels = gmm.fit_predict(X_use)
                s = float(silhouette_score(X_use, labels)) if len(np.unique(labels)) > 1 else float("nan")

                run = {"method": f"GMM_{cov}", "k": k, "silhouette": s, **pca_info}
                all_runs.append(run)

                if not np.isnan(s) and (best_gmm is None or s > best_gmm["silhouette"]):
                    best_gmm = {**run, "labels": labels}

    runs_df = pd.DataFrame(all_runs).sort_values(["method", "silhouette"], ascending=[True, False])
    return runs_df, best_kmeans, best_agnes, best_gmm


def pretty_print_best(best: Dict, title: str) -> None:
    pca_txt = "No PCA" if best["pca"] is None else f"PCA={best['pca']} (expl.var={best['explained_var']:.3f})"
    print("\n======================================")
    print(f"{title}")
    print("--------------------------------------")
    print(f"Best K: {best['k']}")
    print(f"Best silhouette: {best['silhouette']:.4f}")
    print(f"Dim. reduction: {pca_txt}")
    print("======================================\n")


def save_cluster_and_fried_tables(
    merged: pd.DataFrame,
    labels: np.ndarray,
    cluster_col: str,
    clustered_csv_name: str,
    fried_percent_csv_name: str
) -> None:
    out = merged.copy()
    out[cluster_col] = labels
    out.to_csv(OUT_DIR / clustered_csv_name, sep=";", index=False)

    if "fried" in out.columns:
        counts = out.groupby([cluster_col, "fried"]).size().unstack(fill_value=0)
        perc = counts.div(counts.sum(axis=1), axis=0) * 100.0
        perc.to_csv(OUT_DIR / fried_percent_csv_name, sep=";")



def main() -> None:
    merged = merge_datasets()
    print(f"✅ Merge done: rows={len(merged)} users={merged['part_id'].nunique()}")

    X_df, used_cols, dropped_cols = make_X_merged_only(merged)

    prep = build_preprocessor(X_df.copy())
    X_trans = prep.fit_transform(X_df)
    X_dense = X_trans.toarray() if hasattr(X_trans, "toarray") else X_trans

    # Grid 
    pca_list = [None, 2, 5, 10, 20]
    k_list = [2, 3, 4, 5, 6]

    runs_df, best_kmeans, best_agnes, best_gmm = evaluate_grid_kmeans_agnes_gmm(
        X_dense=X_dense,
        pca_list=pca_list,
        k_list=k_list,
        gmm_cov_types=["full", "diag"],
    )

    runs_path = OUT_DIR / "clustering_runs_merged_kmeans_agnes_gmm.csv"
    runs_df.to_csv(runs_path, index=False)

    save_cluster_and_fried_tables(
        merged,
        best_kmeans["labels"],
        cluster_col="cluster_kmeans",
        clustered_csv_name="clustered_merged_best_kmeans.csv",
        fried_percent_csv_name="cluster_fried_percent_best_kmeans.csv",
    )
    save_cluster_and_fried_tables(
        merged,
        best_agnes["labels"],
        cluster_col="cluster_agnes",
        clustered_csv_name="clustered_merged_best_agnes.csv",
        fried_percent_csv_name="cluster_fried_percent_best_agnes.csv",
    )
    save_cluster_and_fried_tables(
        merged,
        best_gmm["labels"],
        cluster_col="cluster_gmm",
        clustered_csv_name="clustered_merged_best_gmm.csv",
        fried_percent_csv_name="cluster_fried_percent_best_gmm.csv",
    )

    report = {
        "merged_only": True,
        "dropped_columns": dropped_cols,
        "used_columns_count": len(used_cols),
        "best_kmeans": {k: v for k, v in best_kmeans.items() if k != "labels"},
        "best_agnes": {k: v for k, v in best_agnes.items() if k != "labels"},
        "best_gmm": {k: v for k, v in best_gmm.items() if k != "labels"},
        "note": "Best runs selected by highest silhouette across PCA and K grids. "
                "fried excluded from clustering features and used only for interpretation.",
    }
    save_json(report, OUT_DIR / "clustering_report_merged_kmeans_agnes_gmm.json")

    pretty_print_best(best_kmeans, "KMeans (best across grid)")
    pretty_print_best(best_agnes, "AGNES / Agglomerative Ward (best across grid)")
    pretty_print_best(best_gmm, "GMM (best across grid)")

    print(f"✅ Saved grid runs CSV: {runs_path}")
    print("✅ Saved best clustered datasets + fried-percent tables in outputs/")
    print(f"✅ Saved report: {OUT_DIR / 'clustering_report_merged_kmeans_agnes_gmm.json'}")


if __name__ == "__main__":
    main()