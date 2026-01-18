from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "font.size": 11,
})


def load_clustered_outputs() -> Dict[str, pd.DataFrame]:

    files = {
        "KMeans": (OUT_DIR / "clustered_merged_best_kmeans.csv", "cluster_kmeans"),
        "GMM": (OUT_DIR / "clustered_merged_best_gmm.csv", "cluster_gmm"),
        "AGNES": (OUT_DIR / "clustered_merged_best_agnes.csv", "cluster_agnes"),
    }

    data: Dict[str, pd.DataFrame] = {}
    for name, (path, cluster_col) in files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run clustering first to generate the clustered outputs."
            )
        df = pd.read_csv(path, sep=";")
        if cluster_col not in df.columns:
            raise ValueError(f"File {path} does not contain expected column '{cluster_col}'.")
        df["part_id"] = df["part_id"].astype(str).str.strip()
        data[name] = df

    return data


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    bool_cols = X.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

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


def make_feature_matrix_for_pca(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:

    forbidden = [
        "weight_loss",
        "exhaustion_score",
        "gait_speed_slower",
        "grip_strength_abnormal",
        "low_physical_activity",
    ]

    drop_cols = ["part_id"]
    if "fried" in df.columns:
        drop_cols.append("fried")
    drop_cols += [c for c in forbidden if c in df.columns]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    return X, X.columns.tolist()


def compute_pca_2d_coords(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:

    X_df, _ = make_feature_matrix_for_pca(df)
    preprocessor = build_preprocessor(X_df.copy())

    X_trans = preprocessor.fit_transform(X_df)
    X_dense = X_trans.toarray() if hasattr(X_trans, "toarray") else X_trans

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_dense)
    expl = float(np.sum(pca.explained_variance_ratio_))

    coords = pd.DataFrame({
        "part_id": df["part_id"].astype(str).str.strip().values,
        "pca1": X_pca[:, 0],
        "pca2": X_pca[:, 1],
    })
    return coords, expl



def plot_pca_scatter_panels(data: Dict[str, pd.DataFrame]) -> None:
    base_df = data["KMeans"]
    coords, expl = compute_pca_2d_coords(base_df)

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.6))
    fig.suptitle("Clusters Visualized in PCA 2D Space", fontsize=14, y=0.98)

    mapping = {
        "KMeans": "cluster_kmeans",
        "GMM": "cluster_gmm",
        "AGNES": "cluster_agnes",
    }

    for ax, method in zip(axes, ["KMeans", "GMM", "AGNES"]):
        df = data[method].merge(coords, on="part_id", how="inner")
        cluster_col = mapping[method]

        for c in sorted(df[cluster_col].unique()):
            sub = df[df[cluster_col] == c]
            ax.scatter(
                sub["pca1"], sub["pca2"],
                s=14, alpha=0.9, linewidths=0,
                label=str(c)
            )

        ax.set_title(method if method != "AGNES" else "Agglomerative", fontsize=12)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend(title="Cluster", fontsize=9, title_fontsize=9, loc="upper right", frameon=True)

        
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    
    fig.text(0.5, 0.01, f"PCA-2 explained variance (sum): {expl:.3f}", ha="center", fontsize=9)

    plt.tight_layout(rect=[0.01, 0.04, 0.99, 0.92])
    out_path = FIG_DIR / "pca_scatter_clusters.png"
    plt.savefig(out_path, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _colored_boxplot(ax: plt.Axes, groups: List[np.ndarray], labels: List[str], colors: List) -> None:
    bp = ax.boxplot(
        groups,
        tick_labels=labels,
        patch_artist=True,
        showfliers=True
    )
    
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.45)
        patch.set_edgecolor("black")

    
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.3)

    for w in bp["whiskers"]:
        w.set_color("black")
    for c in bp["caps"]:
        c.set_color("black")


def plot_age_boxplots_panels(data: Dict[str, pd.DataFrame]) -> None:
    for method, df in data.items():
        if "age" not in df.columns:
            raise ValueError(f"'age' column not found in {method} dataset. Cannot plot age boxplots.")

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.6))
    fig.suptitle("Age Distribution by Cluster", fontsize=14, y=0.98)

    mapping = {
        "KMeans": "cluster_kmeans",
        "GMM": "cluster_gmm",
        "AGNES": "cluster_agnes",
    }

    for ax, method in zip(axes, ["KMeans", "GMM", "AGNES"]):
        df = data[method].copy()
        cluster_col = mapping[method]

        clusters = sorted(df[cluster_col].unique())
        groups = [df.loc[df[cluster_col] == c, "age"].dropna().values for c in clusters]

        default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if len(default_cycle) == 0:
            default_cycle = ["C0", "C1", "C2", "C3", "C4", "C5"]
        colors = [default_cycle[i % len(default_cycle)] for i in range(len(clusters))]

        _colored_boxplot(ax, groups, [str(c) for c in clusters], colors)

        ax.set_title(method if method != "AGNES" else "AGNES", fontsize=12)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Age")

    plt.tight_layout(rect=[0.01, 0.03, 0.99, 0.92])
    out_path = FIG_DIR / "age_boxplots_by_cluster.png"
    plt.savefig(out_path, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _fried_to_text(series: pd.Series) -> pd.Series:

    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        mapping = {0: "Non-frail", 1: "Pre-frail", 2: "Frail"}
        return s.map(mapping).astype("object")
    s = s.astype(str).str.strip()
    norm = s.str.lower()
    out = pd.Series(index=s.index, dtype="object")
    out.loc[norm.isin(["0", "non-frail", "non frail", "nonfrail"])] = "Non-frail"
    out.loc[norm.isin(["1", "pre-frail", "pre frail", "prefrail"])] = "Pre-frail"
    out.loc[norm.isin(["2", "frail"])] = "Frail"
    out = out.fillna(s)
    return out


def plot_fried_pies_combined(data: Dict[str, pd.DataFrame]) -> None:
    FRIED_ORDER = ["Non-frail", "Pre-frail", "Frail"]
    FRIED_COLORS = ["tab:blue", "tab:orange", "tab:green"]  

    mapping = {
        "KMeans": ("cluster_kmeans", "KMeans"),
        "GMM": ("cluster_gmm", "GMM"),
        "AGNES": ("cluster_agnes", "Agglomerative"),
    }

    for method in ["KMeans", "GMM", "AGNES"]:
        if "fried" not in data[method].columns:
            raise ValueError(f"'fried' column not found in {method} dataset. Cannot plot fried pies.")

    clusters_km = sorted(data["KMeans"][mapping["KMeans"][0]].unique())
    clusters_gmm = sorted(data["GMM"][mapping["GMM"][0]].unique())
    clusters_ag = sorted(data["AGNES"][mapping["AGNES"][0]].unique())

    n_km = len(clusters_km)
    n_gmm = len(clusters_gmm)
    n_ag = len(clusters_ag)

    total_cols = max(n_km + n_gmm, n_ag, 4)

    fig = plt.figure(figsize=(18, 8.5), facecolor="white")
    fig.suptitle("Cluster Composition by fried", fontsize=16, y=0.98)

    gs = fig.add_gridspec(
        2, total_cols,
        left=0.03, right=0.97, top=0.90, bottom=0.06,
        wspace=0.65, hspace=0.65
    )

    def pie_for_cluster(ax: plt.Axes, sub: pd.DataFrame, title: str) -> None:
        fried_txt = _fried_to_text(sub["fried"])
        counts = fried_txt.value_counts().reindex(FRIED_ORDER, fill_value=0)

        ax.pie(
            counts.values,
            labels=counts.index,
            colors=FRIED_COLORS,
            autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
            startangle=90,
            textprops={"fontsize": 10},
        )
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")

    col = 0
    for c in clusters_km:
        ax = fig.add_subplot(gs[0, col])
        df = data["KMeans"]
        cluster_col = mapping["KMeans"][0]
        sub = df[df[cluster_col] == c]
        pie_for_cluster(ax, sub, f"Cluster {c} Composition by fried - KMeans")
        col += 1

    for c in clusters_gmm:
        if col >= total_cols:
            break
        ax = fig.add_subplot(gs[0, col])
        df = data["GMM"]
        cluster_col = mapping["GMM"][0]
        sub = df[df[cluster_col] == c]
        pie_for_cluster(ax, sub, f"Cluster {c} Composition by fried - GMM")
        col += 1

    while col < total_cols:
        ax = fig.add_subplot(gs[0, col])
        ax.axis("off")
        col += 1

    start = max((total_cols - n_ag) // 2, 0)
    for i, c in enumerate(clusters_ag):
        ax = fig.add_subplot(gs[1, start + i])
        df = data["AGNES"]
        cluster_col = mapping["AGNES"][0]
        sub = df[df[cluster_col] == c]
        pie_for_cluster(ax, sub, f"Cluster {c} Composition by fried - Agglomerative")

    for j in range(0, start):
        ax = fig.add_subplot(gs[1, j])
        ax.axis("off")
    for j in range(start + n_ag, total_cols):
        ax = fig.add_subplot(gs[1, j])
        ax.axis("off")

    out_path = FIG_DIR / "fried_pies_combined.png"
    plt.savefig(out_path, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# Main

def main() -> None:
    data = load_clustered_outputs()

    plot_pca_scatter_panels(data)

    plot_age_boxplots_panels(data)

    plot_fried_pies_combined(data)

    print("Visualization completed. Check:", FIG_DIR)


if __name__ == "__main__":
    main()
