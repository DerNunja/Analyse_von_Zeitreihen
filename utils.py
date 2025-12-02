from typing import List
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

# UMAP optional
try:
    import umap  
except ImportError:
    umap = None


def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Castet ausgewählte Spalten robust nach float:
    - String -> strip -> Komma durch Punkt ersetzen -> to_numeric
    """
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.strip().str.replace(",", ".", regex=False),
                errors="coerce",
            )
    return df


def build_segment_meta(df: pd.DataFrame, label_column: str, seg_col: str = "SegmentID") -> pd.DataFrame:
    """
    Baut eine Metatabelle pro SegmentID:
      - segment_id
      - label
      - t_start
      - t_end

    Erwartet:
      - 'timestamp' (numerisch)
      - seg_col (z. B. 'SegmentID') mit int / float IDs
    """
    if "timestamp" not in df.columns or seg_col not in df.columns or label_column not in df.columns:
        return pd.DataFrame(columns=["segment_id", "label", "t_start", "t_end"])

    tmp = df.dropna(subset=[seg_col]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=["segment_id", "label", "t_start", "t_end"])

    tmp[seg_col] = tmp[seg_col].astype(int)

    meta = (
        tmp.groupby(seg_col)
        .agg(
            label=(label_column, "first"),
            t_start=("timestamp", "min"),
            t_end=("timestamp", "max"),
        )
        .reset_index()
        .rename(columns={seg_col: "segment_id"})
    )
    return meta


def add_derivative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt aus timestamp, car0_velocity und wheel_position die abgeleiteten Features:
        - vel_acc, vel_jerk
        - steer_vel, steer_jerk
    hinzu, sofern die entsprechenden Spalten existieren.
    """
    df = df.copy()
    required_cols = ["timestamp", "car0_velocity", "wheel_position"]

    if not all(col in df.columns for col in required_cols):
        return df

    # numerische Casts
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["car0_velocity"] = pd.to_numeric(
        df["car0_velocity"].astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )
    df["wheel_position"] = pd.to_numeric(
        df["wheel_position"].astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )

    df = df.sort_values("timestamp")
    dt = df["timestamp"].diff().replace(0, np.nan)

    df["vel_acc"] = df["car0_velocity"].diff() / dt
    df["vel_jerk"] = df["vel_acc"].diff() / dt

    df["steer_vel"] = df["wheel_position"].diff() / dt
    df["steer_jerk"] = df["steer_vel"].diff() / dt

    df[["vel_acc", "vel_jerk", "steer_vel", "steer_jerk"]] = (
        df[["vel_acc", "vel_jerk", "steer_vel", "steer_jerk"]].fillna(0.0)
    )

    return df


def load_run_with_derivatives(
    uploaded_file,
    driver_name: str,
    label_column: str = "Label",
) -> pd.DataFrame:
    """
    Liest eine CSV (Pfad oder Streamlit-Upload), fügt Ableitungen hinzu und setzt eine driver-Spalte.
    """
    df = pd.read_csv(uploaded_file, low_memory=False)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = add_derivative_features(df)
    df["driver"] = driver_name
    return df


def prepare_label_data_for_plots(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    label: str,
    feature_cols: List[str],
    label_column: str,
    driver_a_name: str,
    driver_b_name: str,
    eps: float,
    min_samples: int,
) -> pd.DataFrame:
    """
    Schneidet beide DataFrames auf das gewünschte Label zu, trimmt auf gleiche Länge,
    konvertiert Features zu numerischen Werten, berechnet relative Zeitachsen und
    führt DBSCAN auf den kombinierten Daten aus.

    Gibt ein kombiniertes DataFrame zurück:
      - driver
      - cluster
      - t_rel
      - alle feature_cols
    """

    seg1 = df1[df1[label_column] == label].copy()
    seg2 = df2[df2[label_column] == label].copy()

    if len(seg1) == 0 or len(seg2) == 0:
        raise ValueError(f"Keine Daten für Label '{label}' in einer der Fahrten.")

    seg1["driver"] = driver_a_name
    seg2["driver"] = driver_b_name

    seg1 = _ensure_numeric(seg1, feature_cols + ["timestamp"])
    seg2 = _ensure_numeric(seg2, feature_cols + ["timestamp"])

    min_len = min(len(seg1), len(seg2))
    seg1 = seg1.iloc[:min_len].reset_index(drop=True)
    seg2 = seg2.iloc[:min_len].reset_index(drop=True)

    # relative Zeit
    if "timestamp" in seg1.columns:
        seg1["t_rel"] = seg1["timestamp"] - seg1["timestamp"].iloc[0]
    else:
        seg1["t_rel"] = np.arange(len(seg1))

    if "timestamp" in seg2.columns:
        seg2["t_rel"] = seg2["timestamp"] - seg2["timestamp"].iloc[0]
    else:
        seg2["t_rel"] = np.arange(len(seg2))

    combined = pd.concat([seg1, seg2], ignore_index=True)

    # DBSCAN
    X = combined[feature_cols].fillna(0.0).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    combined["cluster"] = db.fit_predict(X_scaled)

    return combined


def get_default_feature_sets(df_good: pd.DataFrame, df_bad: pd.DataFrame):
    """
    Liefert drei sinnvolle Featurelisten:
      - cluster_features
      - time_series_features
      - dist_features
    gefiltert auf existierende Spalten.
    """
    default_cluster = [
        "throttle",
        "brakes",
        "wheel_position",
        "car0_velocity",
        "car0_engine_rpm",
        "vel_acc",
        "vel_jerk",
        "steer_vel",
        "steer_jerk",
    ]
    default_ts = ["car0_velocity", "wheel_position", "throttle", "brakes"]
    default_dist = ["vel_jerk", "steer_jerk"]

    cluster_features = [
        c for c in default_cluster if c in df_good.columns and c in df_bad.columns
    ]
    time_series_features = [
        c for c in default_ts if c in df_good.columns and c in df_bad.columns
    ]
    dist_features = [
        c for c in default_dist if c in df_good.columns and c in df_bad.columns
    ]
    return cluster_features, time_series_features, dist_features


def compute_embedding(
    X: np.ndarray,
    method: str,
    tsne_perplexity: float = 30.0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Berechnet ein 2D-Embedding mit PCA, t-SNE oder UMAP.
    """
    if method == "PCA":
        pca = PCA(n_components=2, random_state=random_state)
        return pca.fit_transform(X)

    if method == "t-SNE":
        tsne = TSNE(
            n_components=2,
            perplexity=tsne_perplexity,
            learning_rate="auto",
            init="pca",
            random_state=random_state,
        )
        return tsne.fit_transform(X)

    if method == "UMAP":
        if umap is None:
            raise RuntimeError("UMAP ist nicht installiert (Paket `umap-learn`).")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            random_state=random_state,
        )
        return reducer.fit_transform(X)

    raise ValueError(f"Unbekannte Methode: {method}")


def compute_k_distance_curve(
    X: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Berechnet die sortierte k-Distanz-Kurve (k-th nearest neighbor) für DBSCAN-Tuning.

    Standard-Heuristik:
      - X wird standardisiert
      - für jeden Punkt wird die Distanz zum k-ten Nachbarn berechnet
      - die Distanzen werden aufsteigend sortiert
      - 'Elbow' in dieser Kurve ~ sinnvoller eps-Kandidat
    """
    if X.shape[0] < 2:
        return np.array([])

    # Robustheit: k darf nicht größer als Anzahl Punkte sein
    k = max(2, min(k, X.shape[0]))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)

    # letzte Spalte = Distanz zum k-ten Nachbarn (1. ist der Punkt selbst)
    k_dist = np.sort(distances[:, -1])
    return k_dist


def dbscan_eps_sweep(
    X: np.ndarray,
    eps_values: np.ndarray,
    min_samples: int,
) -> pd.DataFrame:
    """
    Sweep über mehrere eps-Werte bei festem min_samples.
    Gibt für jeden eps:
      - n_clusters (ohne Noise-Cluster -1)
      - n_noise
      - noise_ratio
    zurück.
    """
    if X.shape[0] == 0:
        return pd.DataFrame(columns=["eps", "n_clusters", "n_noise", "noise_ratio"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rows = []
    for eps in eps_values:
        db = DBSCAN(eps=float(eps), min_samples=min_samples).fit(X_scaled)
        labels = db.labels_
        n_clusters = len(set(labels) - {-1})
        n_noise = int(np.sum(labels == -1))
        rows.append(
            {
                "eps": float(eps),
                "n_clusters": int(n_clusters),
                "n_noise": n_noise,
                "noise_ratio": n_noise / len(X_scaled),
            }
        )

    return pd.DataFrame(rows)