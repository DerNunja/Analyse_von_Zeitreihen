import io
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import plotly.express as px

# UMAP optional
try:
    import umap  # Paket: umap-learn
except ImportError:
    umap = None


# --------------------- Helper-Funktionen --------------------- #

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
    Liest eine hochgeladene CSV, fügt Ableitungen hinzu und setzt eine driver-Spalte.
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


# --------------------- Streamlit UI --------------------- #

st.set_page_config(page_title="Fahrtenvergleich – Zeitreihen & Cluster", layout="wide")

st.title("🚗 Fahrstil-Analyse mit Zeitreihen & DBSCAN-Clustering")
st.markdown(
    """
Lade zwei **gelabelte CSVs** (gleicher Kurs, zwei Fahrer) und untersuche:
- Unterschiede in Zeitreihen (Velocity, Lenkwinkel, Gas/Bremse)
- Fahrmuster-Cluster (DBSCAN)
- Verteilungen von Jerk/Lenk-Jerk
- 2D-Embedding mit PCA / t-SNE / UMAP

**Voraussetzung:** Beide CSVs haben eine Spalte `timestamp` (Sekunden) und `Label`.
"""
)

with st.sidebar:
    st.header("⚙️ Einstellungen")

    uploaded_a = st.file_uploader("CSV Fahrer A", type=["csv"], key="csv_a")
    uploaded_b = st.file_uploader("CSV Fahrer B", type=["csv"], key="csv_b")

    driver_a_name = st.text_input("Name Fahrer A", value="good")
    driver_b_name = st.text_input("Name Fahrer B", value="bad")
    label_column = st.text_input("Label-Spalte", value="Label")

    st.markdown("---")
    st.markdown("### DBSCAN-Parameter")
    eps = st.slider("eps", min_value=0.1, max_value=3.0, value=0.5, step=0.1)
    min_samples = st.slider("min_samples", min_value=2, max_value=50, value=5, step=1)

    st.markdown("---")
    st.markdown("### Dimensionalitätsreduktion")
    dimred_method = st.selectbox("Methode", ["PCA", "t-SNE", "UMAP"], index=0)

    tsne_perplexity = 30.0
    umap_n_neighbors = 15
    umap_min_dist = 0.1

    if dimred_method == "t-SNE":
        tsne_perplexity = st.slider("t-SNE Perplexity", 5.0, 50.0, 30.0, 1.0)
    elif dimred_method == "UMAP":
        if umap is None:
            st.warning("UMAP ist nicht installiert (Paket `umap-learn`). Bitte installieren und App neu starten.")
        umap_n_neighbors = st.slider("UMAP n_neighbors", 5, 100, 15, 1)
        umap_min_dist = st.slider("UMAP min_dist", 0.0, 0.99, 0.1, 0.01)


if uploaded_a is None or uploaded_b is None:
    st.info("Bitte in der Sidebar zwei gelabelte CSV-Dateien hochladen.")
    st.stop()

# Daten laden
df_good = load_run_with_derivatives(uploaded_a, driver_name=driver_a_name, label_column=label_column)
df_bad = load_run_with_derivatives(uploaded_b, driver_name=driver_b_name, label_column=label_column)

meta_good = build_segment_meta(df_good, label_column=label_column, seg_col="SegmentID")
meta_bad  = build_segment_meta(df_bad,  label_column=label_column, seg_col="SegmentID")

# gemeinsame Labels
labels_good = set(df_good[label_column].dropna().unique()) if label_column in df_good.columns else set()
labels_bad = set(df_bad[label_column].dropna().unique()) if label_column in df_bad.columns else set()
shared_labels = sorted(labels_good.intersection(labels_bad))

if not shared_labels:
    st.error(f"Keine gemeinsamen Labels gefunden (Spalte '{label_column}').")
    st.stop()

st.sidebar.markdown("---")
label_to_plot = st.sidebar.selectbox("Label auswählen", shared_labels)

seg_good_label = meta_good[meta_good["label"] == label_to_plot]
seg_bad_label  = meta_bad[meta_bad["label"] == label_to_plot]

common_seg_ids = sorted(set(seg_good_label["segment_id"]).intersection(seg_bad_label["segment_id"]))

if not common_seg_ids:
    st.sidebar.warning(f"Keine gemeinsamen Segmente für Label '{label_to_plot}' gefunden.")
    selected_segment_id = None
else:
    selected_segment_id = st.sidebar.selectbox(
        f"Segment für Label '{label_to_plot}'",
        common_seg_ids,
        format_func=lambda sid: f"{label_to_plot} #{sid}",
    )

# Default-Features bestimmen
cluster_default, ts_default, dist_default = get_default_feature_sets(df_good, df_bad)

st.sidebar.markdown("### Feature-Auswahl")

cluster_features = st.sidebar.multiselect(
    "Features für Cluster/Embedding",
    options=sorted(list(set(cluster_default + list(df_good.columns.intersection(df_bad.columns))))),
    default=cluster_default,
)

time_series_features = st.sidebar.multiselect(
    "Features für Zeitreihen",
    options=sorted(list(df_good.columns.intersection(df_bad.columns))),
    default=ts_default,
)

dist_features = st.sidebar.multiselect(
    "Features für Verteilungsplots",
    options=sorted(list(df_good.columns.intersection(df_bad.columns))),
    default=dist_default,
)

if not cluster_features:
    st.error("Bitte mindestens ein Feature für Cluster/Embedding auswählen.")
    st.stop()

# --------------------- Analyse & Plots --------------------- #

st.subheader(f"Analyse für Label: **{label_to_plot}**")

try:
    combined = prepare_label_data_for_plots(
        df1=df_good,
        df2=df_bad,
        label=label_to_plot,
        feature_cols=cluster_features,
        label_column=label_column,
        driver_a_name=driver_a_name,
        driver_b_name=driver_b_name,
        eps=eps,
        min_samples=min_samples,
    )
except ValueError as e:
    st.error(str(e))
    st.stop()

# Cluster-Statistik
cluster_labels = combined["cluster"].unique()
cluster_labels_sorted = sorted([c for c in cluster_labels if c != -1]) + ([-1] if -1 in cluster_labels else [])

st.markdown("### Cluster-Statistik")

cluster_rows = []
for cl in cluster_labels_sorted:
    mask = combined["cluster"] == cl
    total = int(mask.sum())
    drivers = combined.loc[mask, "driver"]
    count_a = int((drivers == driver_a_name).sum())
    count_b = int((drivers == driver_b_name).sum())
    freq_a = count_a / total if total > 0 else 0.0
    freq_b = count_b / total if total > 0 else 0.0
    cluster_rows.append(
        {
            "cluster": cl,
            "count_total": total,
            f"count_{driver_a_name}": count_a,
            f"count_{driver_b_name}": count_b,
            f"freq_{driver_a_name}": round(freq_a, 3),
            f"freq_{driver_b_name}": round(freq_b, 3),
        }
    )

st.dataframe(pd.DataFrame(cluster_rows))

# Tabs für Plots
tab_ts, tab_pca, tab_dist = st.tabs(["📈 Zeitreihen", "🔀 Embedding & Cluster", "📊 Verteilungen"])

with tab_ts:
    st.markdown("#### Zeitreihen pro Manöver (Segment)")

    if selected_segment_id is None:
        st.info("Kein gemeinsames Segment für das aktuelle Label vorhanden.")
    else:
        # Filter auf Label + SegmentID
        seg_good_ts = df_good[
            (df_good[label_column] == label_to_plot) &
            (df_good["SegmentID"].astype("Int64") == selected_segment_id)
        ].copy()
        seg_bad_ts = df_bad[
            (df_bad[label_column] == label_to_plot) &
            (df_bad["SegmentID"].astype("Int64") == selected_segment_id)
        ].copy()

        if seg_good_ts.empty or seg_bad_ts.empty:
            st.warning("Für dieses Segment gibt es in einer der Fahrten keine Daten.")
        else:
            # relative Zeit pro Fahrt
            for seg in (seg_good_ts, seg_bad_ts):
                seg["timestamp"] = pd.to_numeric(seg["timestamp"], errors="coerce")

            seg_good_ts = seg_good_ts.sort_values("timestamp")
            seg_bad_ts  = seg_bad_ts.sort_values("timestamp")

            seg_good_ts["t_rel"] = seg_good_ts["timestamp"] - seg_good_ts["timestamp"].iloc[0]
            seg_bad_ts["t_rel"]  = seg_bad_ts["timestamp"] - seg_bad_ts["timestamp"].iloc[0]

            seg_good_ts["driver"] = driver_a_name
            seg_bad_ts["driver"]  = driver_b_name

            combined_seg = pd.concat([seg_good_ts, seg_bad_ts], ignore_index=True)

            st.markdown(
                f"**Segment {selected_segment_id} – Label '{label_to_plot}'** "
                f"(Fahrer: {driver_a_name} vs {driver_b_name})"
            )

            for feat in time_series_features:
                if feat not in combined_seg.columns:
                    continue

                fig_ts = px.line(
                    combined_seg,
                    x="t_rel",
                    y=feat,
                    color="driver",
                    line_dash="driver",
                    title=f"{feat}",
                )
                fig_ts.update_layout(
                    xaxis_title="Zeit [s] (relativ, pro Fahrt normiert)",
                    yaxis_title=feat,
                )
                st.plotly_chart(fig_ts, use_container_width=True)


with tab_pca:
    st.markdown(f"#### Embedding & Clusterplot ({dimred_method})")
    st.caption("Farbe = Cluster, Symbol = Fahrer")

    X = combined[cluster_features].fillna(0.0).to_numpy()

    try:
        X_emb = compute_embedding(
            X,
            method=dimred_method,
            tsne_perplexity=tsne_perplexity,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
        )
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    df_plot = combined.copy()
    df_plot["emb1"] = X_emb[:, 0]
    df_plot["emb2"] = X_emb[:, 1]
    df_plot["cluster_str"] = df_plot["cluster"].astype(str)

    fig_pca = px.scatter(
        df_plot,
        x="emb1",
        y="emb2",
        color="cluster_str",
        symbol="driver",
        title=f"{dimred_method}-Embedding für Label '{label_to_plot}'",
        hover_data=["driver"] + cluster_features,
    )

    # Aufgeräumte Legende: nur noch Cluster (0,1,2,...) – Fahrer über Symbol
    seen_clusters = set()
    for tr in fig_pca.data:
        # Default-Name ist z.B. "-1,good" bzw. "0,bad" -> nur Cluster-Teil nehmen
        raw_name = tr.name  # z.B. "-1,good"
        cluster_part = raw_name.split(",")[0]
        if cluster_part in seen_clusters:
            tr.showlegend = False
        else:
            tr.name = f"Cluster {cluster_part}"
            seen_clusters.add(cluster_part)

    # Plot etwas „größer“ machen und Marker besser sichtbar
    fig_pca.update_traces(marker=dict(size=8, line=dict(width=0.5, color="black")))
    fig_pca.update_layout(
        xaxis_title="Komponente 1",
        yaxis_title="Komponente 2",
        legend_title_text="Cluster",
        height=800,   # mehr Höhe
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig_pca, use_container_width=True)


with tab_dist:
    st.markdown("#### Verteilungen pro Fahrer")

    st.markdown(
        r"""
**Was ist Jerk?**

- In der Fahrdynamik bezeichnet **Jerk** die **Änderung der Beschleunigung pro Zeit**  
  → also die 3. Ableitung der Position nach der Zeit:  
  $$
  \text{Jerk} = \frac{d a}{d t}
  $$
- In dieser App wird Jerk **diskret** aus den Zeitreihen berechnet:
  - `vel_acc` ≈ Änderung der Längsgeschwindigkeit pro Zeit (Längsbeschleunigung)
  - `vel_jerk` ≈ Änderung dieser Beschleunigung pro Zeit  
  - `steer_vel` ≈ Änderung des Lenkwinkels pro Zeit
  - `steer_jerk` ≈ Änderung dieser Lenkgeschwindigkeit pro Zeit

**Was sagt Jerk über den Fahrer aus?**

- **Hohe Beträge** (große positive/negative Werte) heißen:
  - abrupteres Gasgeben/Bremsen (**Längs-Jerk**)  
  - hektischere Lenkkorrekturen (**Lenk-Jerk**)
- **Niedrige Beträge** und eine **eng gebündelte Verteilung** deuten auf:
  - sanftere, vorausschauende Manöver
  - weniger „Zittern“ im Gas/Bremse/Lenken

**Wie liest man die Violin-Plots?**

- X-Achse: Fahrer (z. B. *good* vs. *bad*)
- Y-Achse: Wert des jeweiligen Features (z. B. `vel_jerk`)
- Jede „Violine“ zeigt:
  - die **Verteilungsform** (wo viele Punkte liegen, ist die Violine „dicker“)
  - die eingebettete **Box**: Median + Quartilsbereich
  - einzelne Punkte: die gemessenen Werte (Frames) im gewählten Label
- Vergleiche also:
  - **Lage der Verteilungen** (verschobener Median → systematisch „ruckeliger“)
  - **Breite** (breiter = variabler / weniger konstant)
  - **Ausreißer** (extrem ruppige Einzelereignisse)
"""
    )

    for feat in dist_features:
        if feat not in combined.columns:
            continue

        # Schönere Achsentitel je nach Feature
        if feat == "vel_jerk":
            y_label = "Längs-Jerk (ΔBeschleunigung / Δt)"
            nice_title = "Längs-Jerk (vel_jerk)"
        elif feat == "steer_jerk":
            y_label = "Lenk-Jerk (ΔLenkgeschw. / Δt)"
            nice_title = "Lenk-Jerk (steer_jerk)"
        elif feat == "vel_acc":
            y_label = "Längsbeschleunigung (Δv / Δt)"
            nice_title = "Längsbeschleunigung (vel_acc)"
        elif feat == "steer_vel":
            y_label = "Lenkgeschwindigkeit (ΔLenkwinkel / Δt)"
            nice_title = "Lenkgeschwindigkeit (steer_vel)"
        else:
            y_label = feat
            nice_title = feat

        st.markdown(f"##### {nice_title}")

        # Optional: kurze Interpretation für die zwei wichtigsten
        if feat == "vel_jerk":
            st.caption(
                "Interpretation: Höhere |vel_jerk|-Werte → ruppigeres Gasgeben/Bremsen. "
                "Vergleiche, welcher Fahrer eine breitere oder nach oben/unten verschobene Verteilung hat."
            )
        elif feat == "steer_jerk":
            st.caption(
                "Interpretation: Höhere |steer_jerk|-Werte → hektischere Lenkkorrekturen. "
                "Ein 'ruhiger' Fahrer hat meist eine schmalere Verteilung um 0."
            )

        fig_violin = px.violin(
            combined,
            x="driver",
            y=feat,
            color="driver",
            box=True,
            points="all",
            title=None,
        )
        fig_violin.update_layout(xaxis_title="Fahrer", yaxis_title=y_label)
        st.plotly_chart(fig_violin, use_container_width=True)

st.success("Analyse abgeschlossen. Passe oben die Slider & Feature-Auswahl an, um Effekte zu sehen.")

