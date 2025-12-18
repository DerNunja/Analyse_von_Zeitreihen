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

def _robust_stats_1d(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(iqr=np.nan, mad=np.nan, abs_med=np.nan, std=np.nan)

    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    med = np.median(x)
    iqr = q3 - q1

    mad = np.median(np.abs(x - med))
    abs_med = np.median(np.abs(x))
    std = float(np.std(x))

    return dict(iqr=float(iqr), mad=float(mad), abs_med=float(abs_med), std=float(std))


def _embedding_spread_2d(emb: np.ndarray) -> dict:
    """
    Spread-Maße für 2D-Embedding (robust-ish).
    """
    if emb is None or len(emb) == 0:
        return dict(area=np.nan, trace=np.nan, med_r=np.nan, iqr_r=np.nan)

    emb = emb[np.all(np.isfinite(emb), axis=1)]
    if emb.shape[0] < 3:
        return dict(area=np.nan, trace=np.nan, med_r=np.nan, iqr_r=np.nan)

    center = np.median(emb, axis=0)
    r = np.sqrt(((emb - center) ** 2).sum(axis=1))

    q1 = np.percentile(r, 25)
    q3 = np.percentile(r, 75)

    cov = np.cov(emb.T)
    trace = float(np.trace(cov))
    det = float(np.linalg.det(cov))
    area = float(np.sqrt(det)) if det > 0 else 0.0

    return dict(area=area, trace=trace, med_r=float(np.median(r)), iqr_r=float(q3 - q1))


def compute_driver_quality_scores(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    label: str,
    label_column: str,
    driver_a_name: str,
    driver_b_name: str,
    feature_cols: List[str],
    dimred_method: str = "PCA",
    tsne_perplexity: float = 30.0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int = 42,
    weights: dict | None = None,
) -> dict:
    """
    Liefert einen Score-Vergleich "wer ist schlechter" für ein Label basierend auf:
      - Feature-Streuung (IQR/MAD/abs_med)
      - Embedding-Streuung (area/trace/med_r/iqr_r)

    Score-Interpretation:
      score_driver = gewichtete Summe( normierte Streu-Kennzahlen )
      schlechter = höherer Score

    Returns:
      {
        "label": ...,
        "scores": {driver_a: ..., driver_b: ...},
        "winner_worse": "...",
        "details": {...}
      }
    """
    if weights is None:
        # Default: Violin/Feature-Streuung wichtiger als Embedding
        weights = {
            "feat_iqr": 1.0,
            "feat_mad": 0.7,
            "feat_abs_med": 0.7,
            "emb_area": 0.5,
            "emb_med_r": 0.5,
        }

    # Segmente je Fahrer für dieses Label
    a = df1[df1[label_column] == label].copy()
    b = df2[df2[label_column] == label].copy()
    if a.empty or b.empty:
        raise ValueError(f"Keine Daten für Label '{label}' in einer der Fahrten.")

    a["driver"] = driver_a_name
    b["driver"] = driver_b_name

    # numeric cast (robust)
    a = _ensure_numeric(a, feature_cols)
    b = _ensure_numeric(b, feature_cols)

    # gleiche Länge für fairen Vergleich
    n = min(len(a), len(b))
    a = a.iloc[:n].reset_index(drop=True)
    b = b.iloc[:n].reset_index(drop=True)

    # ---------- Feature-Streuung ----------
    feat_stats = {}
    for drv, seg in [(driver_a_name, a), (driver_b_name, b)]:
        per_feat = {}
        agg_iqr = []
        agg_mad = []
        agg_abs = []
        for c in feature_cols:
            x = seg[c].to_numpy(dtype=float)
            s = _robust_stats_1d(x)
            per_feat[c] = s
            if np.isfinite(s["iqr"]): agg_iqr.append(s["iqr"])
            if np.isfinite(s["mad"]): agg_mad.append(s["mad"])
            if np.isfinite(s["abs_med"]): agg_abs.append(s["abs_med"])
        feat_stats[drv] = {
            "per_feature": per_feat,
            "mean_iqr": float(np.mean(agg_iqr)) if agg_iqr else np.nan,
            "mean_mad": float(np.mean(agg_mad)) if agg_mad else np.nan,
            "mean_abs_med": float(np.mean(agg_abs)) if agg_abs else np.nan,
        }

    # ---------- Embedding-Streuung ----------
    # Embedding auf kombinierten Daten rechnen, dann pro driver splitten
    combined = pd.concat([a, b], ignore_index=True)
    X = combined[feature_cols].fillna(0.0).to_numpy()

    # standardisieren vor Embedding ist wichtig
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    emb = compute_embedding(
        Xs,
        method=dimred_method,
        tsne_perplexity=tsne_perplexity,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        random_state=random_state,
    )

    emb_a = emb[: len(a)]
    emb_b = emb[len(a) :]

    emb_stats = {
        driver_a_name: _embedding_spread_2d(emb_a),
        driver_b_name: _embedding_spread_2d(emb_b),
    }

    # ---------- Normierung & Score ----------
    # Wir normieren jede Kennzahl auf [0..1] über beide Fahrer (min-max), robust genug für 2 Werte.
    def _minmax(v_a, v_b):
        if not (np.isfinite(v_a) and np.isfinite(v_b)):
            return (np.nan, np.nan)
        lo = min(v_a, v_b)
        hi = max(v_a, v_b)
        if hi - lo < 1e-12:
            return (0.5, 0.5)
        return ((v_a - lo) / (hi - lo), (v_b - lo) / (hi - lo))

    a_iqr, b_iqr = _minmax(feat_stats[driver_a_name]["mean_iqr"], feat_stats[driver_b_name]["mean_iqr"])
    a_mad, b_mad = _minmax(feat_stats[driver_a_name]["mean_mad"], feat_stats[driver_b_name]["mean_mad"])
    a_abs, b_abs = _minmax(feat_stats[driver_a_name]["mean_abs_med"], feat_stats[driver_b_name]["mean_abs_med"])
    a_area, b_area = _minmax(emb_stats[driver_a_name]["area"], emb_stats[driver_b_name]["area"])
    a_mr, b_mr = _minmax(emb_stats[driver_a_name]["med_r"], emb_stats[driver_b_name]["med_r"])

    score_a = (
        weights["feat_iqr"] * (a_iqr if np.isfinite(a_iqr) else 0.0)
        + weights["feat_mad"] * (a_mad if np.isfinite(a_mad) else 0.0)
        + weights["feat_abs_med"] * (a_abs if np.isfinite(a_abs) else 0.0)
        + weights["emb_area"] * (a_area if np.isfinite(a_area) else 0.0)
        + weights["emb_med_r"] * (a_mr if np.isfinite(a_mr) else 0.0)
    )
    score_b = (
        weights["feat_iqr"] * (b_iqr if np.isfinite(b_iqr) else 0.0)
        + weights["feat_mad"] * (b_mad if np.isfinite(b_mad) else 0.0)
        + weights["feat_abs_med"] * (b_abs if np.isfinite(b_abs) else 0.0)
        + weights["emb_area"] * (b_area if np.isfinite(b_area) else 0.0)
        + weights["emb_med_r"] * (b_mr if np.isfinite(b_mr) else 0.0)
    )

    winner_worse = driver_a_name if score_a > score_b else driver_b_name

    return {
        "label": label,
        "scores": {driver_a_name: float(score_a), driver_b_name: float(score_b)},
        "winner_worse": winner_worse,
        "details": {
            "feature_spread": feat_stats,
            "embedding_spread": emb_stats,
            "normalized_components": {
                "mean_iqr": {driver_a_name: a_iqr, driver_b_name: b_iqr},
                "mean_mad": {driver_a_name: a_mad, driver_b_name: b_mad},
                "mean_abs_med": {driver_a_name: a_abs, driver_b_name: b_abs},
                "emb_area": {driver_a_name: a_area, driver_b_name: b_area},
                "emb_med_r": {driver_a_name: a_mr, driver_b_name: b_mr},
            },
            "weights": weights,
        },
    }
