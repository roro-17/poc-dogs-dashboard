import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from utils import load_metrics, extract_models_list, get_confusion_matrix, metrics_to_dataframe

# --- Thème & Palette ---
THEME = {
    "template": "plotly_white",  # 'plotly', 'plotly_white', 'ggplot2', 'seaborn', 'simple_white', ...
    "bars_multi": ["#2563eb", "#22c55e", "#f97316", "#ef4444", "#8b5cf6", "#14b8a6"],  # pour les graphes à >1 métrique
    "bar_single": "#2563eb",  # pour un seul indicateur
    "cm_scale": "Blues",  # palette de la matrice de confusion (ex: 'Viridis', 'Cividis', 'Blues', 'YlOrBr')
}

# Couleurs CONSISTENTES par métrique (recommandé)
COLOR_MAP = {
    "top1_accuracy": "#2563eb",
    "top5_accuracy": "#f97316",
    "macro_f1": "#16a34a",
    "test_loss": "#ef4444",
    "eval_time_sec": "#8b5cf6",
    "model_size_mb": "#10b981",
    "train_time_min": "#a855f7",
    "training_time_min": "#a855f7",
}

st.set_page_config(page_title="Dogs PoC Dashboard", layout="wide")
plt.style.use("ggplot")
st.markdown(
    """
    <style>
      .main .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
      h1, h2, h3 {color: #1f4f8b;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🐶 Stanford Dogs — PoC Dashboard")

METRICS = os.getenv("METRICS_PATH", "/app/app/data/metrics.json")


# ---------- Helpers ----------
def bar_multi(df, metrics, title, ylabel="Value", legend_title="Metric"):
    used = [m for m in metrics if m in df.columns]
    if not used:
        st.info(f"Métriques absentes: {metrics}")
        return

    long_df = df.melt(id_vars=["model"], value_vars=used,
                      var_name="Metric", value_name="Value")

    # Plusieurs métriques -> couleurs par Metric
    if len(used) > 1:
        fig = px.bar(
            long_df, x="model", y="Value", color="Metric", barmode="group",
            color_discrete_map=COLOR_MAP,  # couleurs stables par nom de métrique
            hover_data={"Value": ":.3f"}
        )
        fig.update_layout(legend_title_text=legend_title)

    # Une seule métrique -> couleur unique
    else:
        color = COLOR_MAP.get(used[0], THEME["bar_single"])
        fig = px.bar(
            long_df, x="model", y="Value",
            color_discrete_sequence=[color],
            hover_data={"Value": ":.3f"}
        )
        fig.update_layout(legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center", yanchor="bottom"),
                          legend_title_text="")

    # Mise en forme commune
    fig.update_layout(
        template=THEME["template"],
        title=title,
        xaxis_title="model",
        yaxis_title=ylabel,
        margin=dict(t=80, l=40, r=20, b=60),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center", yanchor="bottom"),
    )
    st.plotly_chart(fig, use_container_width=True)


metrics_obj = load_metrics(METRICS)
if not metrics_obj:
    st.warning("Aucun métrique trouvé. Dépose un fichier metrics.json dans app/data.")
    st.stop()

df = metrics_to_dataframe(metrics_obj)
if df.empty:
    st.error("Le format de metrics.json n'est pas reconnu. Attendu: liste de dicts ou {'models': [...]}")
    st.stop()

with st.expander("ℹ️ Description des métriques", expanded=True):
    st.markdown(
        """
- **Loss** : Cross-Entropy (NLL moyenne).
- **Top-1 accuracy** : précision stricte (classe vraie = prédiction la plus probable).
- **Top-5 accuracy** : correcte si la classe vraie est dans les 5 meilleures prédictions.
- **Macro F1-score** : F1 moyen non pondéré sur toutes les classes (utile en cas de déséquilibre).
- **Temps d’entraînement (min)** : durée d’apprentissage (si présente).
- **Temps d’évaluation (s)** : durée d’inférence sur le test set.
- **Taille du modèle (Mo)** : paramètres + buffers.
        """
    )

st.subheader("📊 Comparaison des modèles")
st.dataframe(df, use_container_width=True)

# ---------- Charts ----------
c1, c2 = st.columns(2)
with c1:
    bar_multi(df, ["top1_accuracy", "top5_accuracy"],
              "Comparaison des précisions (Top-1 / Top-5)")
with c2:
    bar_multi(df, ["macro_f1", "test_loss"],
              "Macro-F1 vs Loss", ylabel="Score")

# ---------- Temps & Taille ----------
st.subheader("⏱️ Temps & 📦 Taille")

# Tableau récap (affiche ce qui est disponible)
cols_time_size = ["model"]
if "eval_time_sec" in df.columns: cols_time_size.append("eval_time_sec")
if "model_size_mb" in df.columns: cols_time_size.append("model_size_mb")
if "train_time_min" in df.columns: cols_time_size.append("train_time_min")
if "training_time_min" in df.columns and "training_time_min" not in cols_time_size:
    cols_time_size.append("training_time_min")

if len(cols_time_size) > 1:
    st.dataframe(df[cols_time_size], use_container_width=True)
else:
    st.info("Pas de colonnes temps/taille dans metrics.json")

# Deux graphes côte à côte
g1, g2 = st.columns(2)
with g1:
    if "eval_time_sec" in df.columns:
        bar_multi(df, ["eval_time_sec"], "Temps d’évaluation (secondes)",
                  ylabel="Secondes", legend_title="")
    else:
        st.info("Pas de 'eval_time_sec' dans metrics.json.")
with g2:
    if "model_size_mb" in df.columns:
        bar_multi(df, ["model_size_mb"], "Taille du modèle (Mo)",
                  ylabel="Mo", legend_title="")
    else:
        st.info("Pas de 'model_size_mb' dans metrics.json.")

models = extract_models_list(metrics_obj)
if models:
    st.subheader("🧩 Matrice de confusion")
    selected = st.selectbox("Modèle :", models, index=0)
    cm = get_confusion_matrix(metrics_obj, selected)
    if cm is None:
        st.info("Aucune 'confusion_matrix' trouvée pour ce modèle dans metrics.json.")
    else:
        normalize = st.checkbox("Normaliser par lignes", value=True)
        cm_plot = np.array(cm, dtype=float)
        if normalize:
            row_sums = cm_plot.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_plot = cm_plot / row_sums

        fig = px.imshow(
            cm_plot,
            color_continuous_scale=THEME["cm_scale"],  # << change juste ici pour la palette
            aspect="auto",
            origin="upper",
            labels=dict(color="Intensité")
        )
        fig.update_layout(template=THEME["template"], margin=dict(t=40, l=40, r=20, b=40))
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Aucun nom de modèle détecté dans metrics.json ('model').")
