
from pathlib import Path
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_data() -> pd.DataFrame:
    merged_dataset_path = OUT_DIR / "merged_dataset.csv"
    if not merged_dataset_path.exists():
        raise FileNotFoundError(
            f"Could not find {merged_dataset_path}. Run preprocessing first to generate merged_dataset.csv."
        )

    return pd.read_csv(merged_dataset_path, sep=";")


def apply_gaussian_mxture(df: pd.DataFrame, pca: bool = False, n_PCA_components: float = 20) -> pd.DataFrame:
    scaler = StandardScaler()
    features = df.columns#.drop('fried')

    X = df[features]

    X_scaled = scaler.fit_transform(X)
    gmm = GaussianMixture(n_components=3, random_state=42)

    if pca:
        X_pca = PCA(n_components = n_PCA_components).fit_transform(X_scaled)
        labels = gmm.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)

    else:
        labels = gmm.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)


    df['cluster'] = labels
    print('silhouette: ', score)
    return df.copy()


def apply_kmeans(df: pd.DataFrame, pca: bool = False, n_PCA_components: float = 20) -> pd.DataFrame: #High silhouette but grouped by cluster which is wrong

    importance_dict = {
        "gait_speed_4m": 0.08191764550780753,
        "gait_get_up": 0.05937834220130965,
        "activity_regular": 0.039901544352536775,
        "mmse_total_score": 0.03974943264205652,
        "raise_chair_time": 0.0396977022957149,
        "depression_total_score": 0.0376100871898216,
        "age": 0.03606346082117315,
        "bmi_score": 0.0357611167909896,
        "waist": 0.034931455179306535,
        "pain_perception": 0.03428244263991674,
        "anxiety_perception": 0.033846290677894096,
        "leisure_out": 0.03297523467804235,
        "social_text": 0.03246536743856863,
        "comorbidities_count": 0.028449744698221525,
        "cognitive_total_score": 0.026816714470748498,
        "bmi_body_fat": 0.026122520565916164,
        "social_phone": 0.025193383749068373,
        "screening_score": 0.02497417257093041,
        "lean_body_mass": 0.024905617168632866,
        "iadl_grade": 0.024533538660827636,
        "medication_count": 0.02445561098211231,
        "life_quality": 0.024192901539513756,
        "social_calls": 0.02349358212800587,
        "alcohol_units": 0.01884907157913644,
        "stairs_number": 0.018763609442389893,
        "health_rate": 0.018509135823066146,
        "social_visits": 0.018016945564924374,
        "health_rate_comparison": 0.012056051774139275,
        "hospitalization_three_years": 0.011184981278542885,
        "leisure_club": 0.011013839420244935,
        "comorbidities_significant_count": 0.010174673635537108,
        "balance_single": 0.008740108990475403,
        "falls_one_year": 0.008173132166098218,
        "audition": 0.008015380568044538,
        "smoking": 0.007366327142947264,
        "sleep": 0.007314482868674014,
        "vision": 0.006787328273480987,
        "hospitalization_one_year": 0.006529464194340784,
        "living_alone": 0.005663559937988299,
        "katz_index": 0.00517281302784413,
        "ortho_hypotension": 0.004344928606490415,
        "gender": 0.004233364295425096,
        "memory_complain": 0.00389586138855918,
        "fractures_three_years": 0.0036552539292345793,
        "social_skype": 0.0031780679680362754,
        "gait_optional_binary": 0.002920068463836369,
        "house_suitable_professional": 0.002007403087712348,
        "house_suitable_participant": 0.0017162376237156046
    }

    features = df.columns.drop('fried')

    X = df[features]

    X_weighted = X.copy()

    for col in X.columns:
        if col in importance_dict:
            X_weighted[col] = X[col] * importance_dict[col]
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )
    if pca:
        X_pca = PCA(n_components = n_PCA_components).fit_transform(X)
        labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)


    else:
        labels = kmeans.fit_predict(X_weighted)
        score = silhouette_score(X_weighted, labels)


    df["cluster"] = labels

    print('silhouette: ', score)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
    return df.copy()



def apply_kmeans_normalize_and_multiply_importance(df: pd.DataFrame, pca: bool = False, n_PCA_components: float = 20) -> pd.DataFrame:

    importance_dict = {
        "gait_speed_4m": 0.08191764550780753,
        "gait_get_up": 0.05937834220130965,
        "activity_regular": 0.039901544352536775,
        "mmse_total_score": 0.03974943264205652,
        "raise_chair_time": 0.0396977022957149,
        "depression_total_score": 0.0376100871898216,
        "age": 0.03606346082117315,
        "bmi_score": 0.0357611167909896,
        "waist": 0.034931455179306535,
        "pain_perception": 0.03428244263991674,
        "anxiety_perception": 0.033846290677894096,
        "leisure_out": 0.03297523467804235,
        "social_text": 0.03246536743856863,
        "comorbidities_count": 0.028449744698221525,
        "cognitive_total_score": 0.026816714470748498,
        "bmi_body_fat": 0.026122520565916164,
        "social_phone": 0.025193383749068373,
        "screening_score": 0.02497417257093041,
        "lean_body_mass": 0.024905617168632866,
        "iadl_grade": 0.024533538660827636,
        "medication_count": 0.02445561098211231,
        "life_quality": 0.024192901539513756,
        "social_calls": 0.02349358212800587,
        "alcohol_units": 0.01884907157913644,
        "stairs_number": 0.018763609442389893,
        "health_rate": 0.018509135823066146,
        "social_visits": 0.018016945564924374,
        "health_rate_comparison": 0.012056051774139275,
        "hospitalization_three_years": 0.011184981278542885,
        "leisure_club": 0.011013839420244935,
        "comorbidities_significant_count": 0.010174673635537108,
        "balance_single": 0.008740108990475403,
        "falls_one_year": 0.008173132166098218,
        "audition": 0.008015380568044538,
        "smoking": 0.007366327142947264,
        "sleep": 0.007314482868674014,
        "vision": 0.006787328273480987,
        "hospitalization_one_year": 0.006529464194340784,
        "living_alone": 0.005663559937988299,
        "katz_index": 0.00517281302784413,
        "ortho_hypotension": 0.004344928606490415,
        "gender": 0.004233364295425096,
        "memory_complain": 0.00389586138855918,
        "fractures_three_years": 0.0036552539292345793,
        "social_skype": 0.0031780679680362754,
        "gait_optional_binary": 0.002920068463836369,
        "house_suitable_professional": 0.002007403087712348,
        "house_suitable_participant": 0.0017162376237156046
    }

    

    features = df.columns.drop('fried')

    X = df[features]

    scaler = StandardScaler()
    X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    max_imp = max(importance_dict.values())
    importance_norm = {k: v / max_imp for k, v in importance_dict.items()}

    X_weighted = X_normalized.copy()
    for col in X_weighted.columns:
        if col in importance_norm:
            X_weighted[col] = X_weighted[col] * importance_norm[col] * 10
    
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )
    if pca:
        X_pca = PCA(n_components = n_PCA_components).fit_transform(X)
        labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)


    else:
        labels = kmeans.fit_predict(X_weighted)
        score = silhouette_score(X_weighted, labels)


    df["cluster"] = labels

    print('silhouette: ', score)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df[["fried", "cluster"]])
    return df.copy()


def apply_agglomerative(df: pd.DataFrame, pca: bool = False, n_PCA_components: float = 20):

    scaler = StandardScaler()

    features = df.columns.drop('fried')

    X = df[features]

    X_scaled = scaler.fit_transform(X)
    clusterer = AgglomerativeClustering(n_clusters=3, linkage='ward')

    if pca:
        X_pca = PCA(n_components = n_PCA_components).fit_transform(X_scaled)
        labels = clusterer.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
    else:
        labels = clusterer.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)


    df["cluster"] = labels

    score = silhouette_score(X_scaled, labels)

    print('silhouette: ', score)


def apply_spectral_clustering(df: pd.DataFrame, pca: bool = False, n_PCA_components: float = 20):

    scaler = StandardScaler()
    features = df.columns.drop('fried')

    X = df[features]

    X_scaled = scaler.fit_transform(X)

    sc = SpectralClustering(
        n_clusters=3,
        affinity="nearest_neighbors",  # or "rbf"
        n_neighbors=10,
        assign_labels="kmeans",
        random_state=42
    )
    if pca:
        X_pca = PCA(n_components = n_PCA_components).fit_transform(X_scaled)
        labels = sc.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
    else:
        labels = (X_scaled)
        score = silhouette_score(X_scaled, labels)
    df["cluster"] = labels

    score = silhouette_score(X_scaled, labels)

    print('silhouette: ', score)


if __name__ == "__main__":
    df = read_data()
    apply_kmeans(df, True, 20)
    #apply_gaussian_mxture(df, True)
    #apply_agglomerative(df)