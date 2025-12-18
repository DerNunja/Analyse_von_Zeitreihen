import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

try:
    import umap  # type: ignore
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from utils import (
    load_run_with_derivatives,
    build_segment_meta,
    get_default_feature_sets,
    prepare_label_data_for_plots,
    compute_embedding,
    compute_k_distance_curve,   
    dbscan_eps_sweep,   
    compute_driver_quality_scores,        
)



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
        if not HAS_UMAP:
            st.warning("UMAP ist nicht installiert (Paket `umap-learn`). Bitte installieren und App neu starten.")
        umap_n_neighbors = st.slider("UMAP n_neighbors", 5, 100, 15, 1)
        umap_min_dist = st.slider("UMAP min_dist", 0.0, 0.99, 0.1, 0.01)


if uploaded_a is None or uploaded_b is None:
    st.info("Bitte in der Sidebar zwei gelabelte CSV-Dateien hochladen.")
    st.stop()

# Daten laden
df_good = load_run_with_derivatives(uploaded_a, driver_a_name, label_column=label_column)
df_bad = load_run_with_derivatives(uploaded_b, driver_b_name, label_column=label_column)

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
tab_ts, tab_pca, tab_dist, tab_score = st.tabs(["📈 Zeitreihen", "🔀 Embedding & Cluster", "📊 Verteilungen", "🏁 Bewertung"])

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

    # Farbe = Fahrer (A = grün, B = rot), Symbol = Cluster-ID
    fig_pca = px.scatter(
        df_plot,
        x="emb1",
        y="emb2",
        color="driver",
        symbol="cluster_str",
        color_discrete_map={
            driver_a_name: "green",
            driver_b_name: "red",
        },
        title=f"{dimred_method}-Embedding für Label '{label_to_plot}'",
        hover_data=["driver", "cluster_str"] + cluster_features,
    )

    fig_pca.update_traces(
        marker=dict(
            size=8,
            line=dict(width=0.5, color="black"),
        )
    )
    fig_pca.update_layout(
        xaxis_title="Komponente 1",
        yaxis_title="Komponente 2",
        legend_title_text="Fahrer / Cluster",
        height=800,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.caption(
        f"Farbe = Fahrer (grün = {driver_a_name}, rot = {driver_b_name}), "
        "Symbol = Cluster-ID."
    )

    st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("---")
    st.markdown("### DBSCAN-Tuning (Elbow-Heuristik)")

    st.markdown(
    """
    **Ziel:** sinnvolle Wahl von `eps` und `min_samples` für DBSCAN.

    - **k-Distanz-Plot** (links):  
    sortierte Distanz zum k-ten Nachbarn.  
    Ein „Knick“ (Elbow) in der Kurve ist ein guter Kandidat für `eps`.
    - **Cluster vs. eps** (rechts):  
    zeigt, wie sich die Anzahl Cluster (und Noise) bei variierendem `eps` verhält.
    """
    )

    # --- Parameter für Tuning (lokal, unabhängig von den Slidern oben verwendbar) ---
    col_left, col_right = st.columns(2)

    with col_left:
        k_for_kdist = st.slider(
            "k für k-Distanz-Plot (typisch = min_samples)",
            min_value=2,
            max_value=max(3, min(50, combined.shape[0])),
            value=min_samples,
            step=1,
            key="k_for_kdist",
        )

    with col_right:
        eps_min, eps_max = st.slider(
            "eps-Range für Sweep",
            min_value=0.1,
            max_value=3.0,
            value=(max(0.1, eps - 0.5), min(3.0, eps + 0.5)),
            step=0.1,
            key="eps_sweep_range",
        )

    # --- Datenbasis für Tuning (ohne erneuten DBSCAN) ---
    X_tune = combined[cluster_features].fillna(0.0).to_numpy()

    # k-Distanz-Plot
    with col_left:
        k_dist = compute_k_distance_curve(X_tune, k=k_for_kdist)
        if k_dist.size == 0:
            st.info("Zu wenige Punkte für k-Distanz-Plot.")
        else:
            idx = np.arange(len(k_dist))
            fig_k = px.line(
                x=k_dist,
                y=idx,
                labels={"x": "Punkte (sortiert)", "y": f"{k_for_kdist}-Distanz"},
                title="k-Distanz-Plot",
            )
            fig_k.update_layout(margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_k, use_container_width=True)

    # eps-Sweep-„Elbow“-Plot
    with col_right:
      

        eps_values = np.linspace(eps_min, eps_max, num=20)
        df_sweep = dbscan_eps_sweep(X_tune, eps_values, min_samples=min_samples)

        if df_sweep.empty:
            st.info("Keine Daten für eps-Sweep verfügbar.")
        else:
            fig_eps = px.line(
                df_sweep,
                x="eps",
                y="n_clusters",
                markers=True,
                labels={"eps": "eps", "n_clusters": "Anzahl Cluster"},
                title="Cluster-Anzahl vs. eps",
            )
            fig_eps.update_layout(margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_eps, use_container_width=True)

            # optional: zweite Info-Tabelle
            with st.expander("Details zum eps-Sweep (inkl. Noise-Anteil)"):
                st.dataframe(
                    df_sweep[["eps", "n_clusters", "n_noise", "noise_ratio"]]
                    .style.format({"noise_ratio": "{:.2%}"})
                )


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

with tab_score:
    st.markdown("## 🏁 Computergestützte Fahrerbewertung")

    st.markdown(
        """
Die Bewertung basiert auf der Hypothese:

- **Mehr Streuung / mehr Extremwerte** in wichtigen Fahrdynamik-Features (z. B. Jerk)  
  → deutet auf **unsicherere bzw. weniger konstante** Fahrweise hin.

Wir messen Streuung robust mit **IQR** und **MAD** sowie “Ruppigkeit” mit **median(|x|)**.
Zusätzlich wird die **Streuung im 2D-Embedding** (PCA/t-SNE/UMAP) berücksichtigt.
"""
    )

    # Welche Features sollen in den Score eingehen?
    score_features = st.multiselect(
        "Features für die Bewertung",
        options=sorted(list(df_good.columns.intersection(df_bad.columns))),
        default=[c for c in ["vel_jerk", "steer_jerk", "vel_acc", "steer_vel"] if c in df_good.columns and c in df_bad.columns],
    )

    if not score_features:
        st.warning("Bitte mindestens ein Feature auswählen.")
        st.stop()

    # Gewichtung als Slider (optional)
    with st.expander("Gewichte anpassen"):
        w_feat_iqr = st.slider("Gewicht: Feature-IQR", 0.0, 2.0, 1.0, 0.1)
        w_feat_mad = st.slider("Gewicht: Feature-MAD", 0.0, 2.0, 0.7, 0.1)
        w_feat_abs = st.slider("Gewicht: median(|x|)", 0.0, 2.0, 0.7, 0.1)
        w_emb_area = st.slider("Gewicht: Embedding-Fläche", 0.0, 2.0, 0.5, 0.1)
        w_emb_medr = st.slider("Gewicht: Embedding median radius", 0.0, 2.0, 0.5, 0.1)

    weights = {
        "feat_iqr": w_feat_iqr,
        "feat_mad": w_feat_mad,
        "feat_abs_med": w_feat_abs,
        "emb_area": w_emb_area,
        "emb_med_r": w_emb_medr,
    }

    # Score für aktuelles Label
    try:
        res = compute_driver_quality_scores(
            df1=df_good,
            df2=df_bad,
            label=label_to_plot,
            label_column=label_column,
            driver_a_name=driver_a_name,
            driver_b_name=driver_b_name,
            feature_cols=score_features,
            dimred_method=dimred_method,
            tsne_perplexity=tsne_perplexity,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            weights=weights,
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.markdown(f"### Ergebnis für Label **{label_to_plot}**")
    sA = res["scores"][driver_a_name]
    sB = res["scores"][driver_b_name]
    worse = res["winner_worse"]

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Score {driver_a_name}", f"{sA:.3f}")
    c2.metric(f"Score {driver_b_name}", f"{sB:.3f}")
    c3.metric("Schlechter (höherer Score)", worse)

    with st.expander("Details (Feature- & Embedding-Streuung)"):
        st.json(res["details"])

    # Optional: Overall über alle Labels
    st.markdown("---")
    st.markdown("### Overall über alle Labels")

    do_overall = st.checkbox("Overall-Score über alle gemeinsamen Labels berechnen", value=False)
    if do_overall:
        rows = []
        for lbl in shared_labels:
            try:
                rr = compute_driver_quality_scores(
                    df1=df_good,
                    df2=df_bad,
                    label=lbl,
                    label_column=label_column,
                    driver_a_name=driver_a_name,
                    driver_b_name=driver_b_name,
                    feature_cols=score_features,
                    dimred_method=dimred_method,
                    tsne_perplexity=tsne_perplexity,
                    umap_n_neighbors=umap_n_neighbors,
                    umap_min_dist=umap_min_dist,
                    weights=weights,
                )
                rows.append({
                    "label": lbl,
                    f"score_{driver_a_name}": rr["scores"][driver_a_name],
                    f"score_{driver_b_name}": rr["scores"][driver_b_name],
                    "worse": rr["winner_worse"],
                })
            except Exception:
                continue

        df_scores = pd.DataFrame(rows)
        st.dataframe(df_scores)

        if not df_scores.empty:
            mean_a = df_scores[f"score_{driver_a_name}"].mean()
            mean_b = df_scores[f"score_{driver_b_name}"].mean()
            worse_overall = driver_a_name if mean_a > mean_b else driver_b_name

            st.markdown("#### Zusammenfassung")
            st.write(f"Ø Score {driver_a_name}: **{mean_a:.3f}**")
            st.write(f"Ø Score {driver_b_name}: **{mean_b:.3f}**")
            st.write(f"➡️ Insgesamt schlechter: **{worse_overall}**")
