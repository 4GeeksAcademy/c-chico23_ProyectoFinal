"""
streamlit_app.py
Aplicación Streamlit — Análisis del Mercado Musical con Last.fm

Ejecutar desde la raíz del proyecto:
    streamlit run streamlit_app.py
"""

# ── Librerías ─────────────────────────────────────────────────────────────────
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

# Ruta al CSV procesado (ya tiene todas las features calculadas del notebook)
_BASE     = os.path.dirname(os.path.abspath(__file__))
_DATA_CSV = os.path.join(_BASE, "src", "data", "df_clean.csv")


@st.cache_data(show_spinner="Cargando datos...")
def load_data() -> pd.DataFrame:
    """
    Carga df_clean.csv — el dataset limpio con todas las features ya calculadas.
    Se usa @st.cache_data para no releer el archivo en cada interacción.
    """
    df = pd.read_csv(_DATA_CSV, low_memory=False)
    for col in ["playcount", "listeners"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CARGA DEL MODELO
# ═══════════════════════════════════════════════════════════════════════════════

_MODELS_DIR = os.path.join(_BASE, "src", "models")


@st.cache_resource(show_spinner="Cargando modelo...")
def load_model():
    """
    Carga los tres artefactos del modelo entrenado en el notebook:
      - modelo_hit.pkl   → RandomForestClassifier
      - encoder_tag.pkl  → LabelEncoder (convierte género texto → número)
      - features.pkl     → lista con el orden exacto de columnas que espera el modelo
    Se usa @st.cache_resource para cargar una sola vez por sesión.
    """
    model    = joblib.load(os.path.join(_MODELS_DIR, "modelo_hit.pkl"))
    encoder  = joblib.load(os.path.join(_MODELS_DIR, "encoder_tag.pkl"))
    features = joblib.load(os.path.join(_MODELS_DIR, "features.pkl"))
    return model, encoder, features


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PREDICCIÓN
# ═══════════════════════════════════════════════════════════════════════════════

# Umbral p90 calculado sobre backup_tracks.csv en el notebook
# Una canción es "hit" si supera este número de reproducciones
P90_THRESHOLD = 5_079_698


def predecir_hit(model, encoder, features, duracion_min, genero, oyentes, engagement):
    """
    Prepara el vector de entrada y devuelve la probabilidad de hit.

    El modelo fue entrenado con 9 columnas, incluyendo log_playcount e is_hit.
    Para una canción nueva las derivamos desde los inputs del usuario:
      - log_playcount = log(1 + oyentes × engagement)
      - is_hit = 1 si el playcount estimado supera el umbral p90

    Devuelve un diccionario con probability, label, emoji e is_short.
    """
    # Convertir género de texto a número con el encoder guardado
    tag_enc = encoder.transform([genero])[0] if genero in encoder.classes_ else 0

    # Estimar el playcount a partir de los inputs del usuario
    playcount_estimado = oyentes * engagement
    log_play_est       = np.log1p(playcount_estimado)
    is_hit_est         = int(playcount_estimado >= P90_THRESHOLD)

    # Crear la fila de entrada con todas las columnas en el orden correcto
    fila = pd.DataFrame([{
        "log_listeners"          : np.log1p(oyentes),
        "duration_min"           : duracion_min,
        "is_short_track"         : int(duracion_min < 2.5),
        "tag_encoded"            : tag_enc,
        "artist_track_count"     : 1,
        "track_share_of_artist"  : 1.0,
        "playcount_per_listener" : engagement,
        "log_playcount"          : log_play_est,
        "is_hit"                 : is_hit_est,
    }]).reindex(columns=features, fill_value=0)

    # Predecir — ignorar warnings de sklearn sobre nombres de columnas
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob = model.predict_proba(fila)[0][1] * 100

    # Clasificar el resultado
    if prob >= 70:
        label, emoji = "Hit potencial", "🚀"
    elif prob >= 45:
        label, emoji = "Potencial medio", "🟡"
    else:
        label, emoji = "Bajo potencial", "📉"

    return {
        "probability" : round(prob, 1),
        "label"       : label,
        "emoji"       : emoji,
        "is_short"    : duracion_min < 2.5,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════════

sns.set_theme(style="whitegrid", font_scale=1.0)


def plot_top_tracks(df, n=15):
    """
    Barplot horizontal con las top N canciones por reproducciones.
    Devuelve una Figure de matplotlib (sin hacer plt.show).
    """
    top = (
        df[["name", "artist", "playcount"]]
        .dropna(subset=["playcount"])
        .sort_values("playcount", ascending=False)
        .head(n)
        .copy()
    )
    top["label"] = top["name"] + "  —  " + top["artist"]
    top = top.sort_values("playcount", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top["label"], top["playcount"] / 1e6, color="steelblue", edgecolor="white")

    ax.set_xlabel("Reproducciones (millones)", fontsize=11)
    ax.set_title(f"🎵 Top {n} canciones por reproducciones", fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))

    for bar, val in zip(bars, top["playcount"] / 1e6):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}M", va="center", fontsize=8.5, color="dimgray")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_genres(df, n=12):
    """
    Barplot de géneros por reproducciones totales.
    Devuelve una Figure de matplotlib.
    """
    df_tag = df.dropna(subset=["tag", "playcount"]).copy()

    if df_tag.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Sin datos de género disponibles",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
        return fig

    stats = (
        df_tag.groupby("tag")["playcount"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
        .sort_values("playcount", ascending=True)
    )

    colors = sns.color_palette("Blues_r", n_colors=len(stats))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(stats["tag"], stats["playcount"] / 1e9, color=colors, edgecolor="white")

    ax.set_xlabel("Reproducciones totales (miles de millones)", fontsize=11)
    ax.set_title(f"🎧 Top {n} géneros por popularidad", fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}B"))

    for bar, val in zip(bars, stats["playcount"] / 1e9):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}B", va="center", fontsize=8.5, color="dimgray")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_heatmap(df):
    """
    Heatmap de reproducciones medias por Género × País.
    Solo incluye países reales (excluye GLOBAL y UNKNOWN).
    Devuelve una Figure de matplotlib.
    """
    df_geo = df[
        df["country"].notna() &
        ~df["country"].isin(["UNKNOWN", "GLOBAL"]) &
        df["tag"].notna()
    ].copy()

    if df_geo.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Sin datos geográficos disponibles",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
        return fig

    top_tags = (df_geo.groupby("tag")["playcount"].sum()
                .sort_values(ascending=False).head(10).index.tolist())
    top_countries = (df_geo.groupby("country")["playcount"].sum()
                     .sort_values(ascending=False).head(10).index.tolist())

    df_geo = df_geo[df_geo["tag"].isin(top_tags) & df_geo["country"].isin(top_countries)]

    pivot = (df_geo.groupby(["tag", "country"])["playcount"]
             .mean().unstack(fill_value=0) / 1e6)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(pivot, ax=ax, cmap="YlGn", annot=True, fmt=".1f",
                linewidths=0.4, linecolor="white",
                cbar_kws={"label": "Reproducciones medias (M)", "shrink": 0.8})
    ax.set_title("📊 Reproducciones medias por Género × País (millones)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("País", fontsize=11)
    ax.set_ylabel("Género", fontsize=11)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5. APP STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🎵 Music Hit Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    hr { margin: 1.5rem 0; border: none; border-top: 1px solid #e0e0e0; }
</style>
""", unsafe_allow_html=True)

# Cargar datos y modelo
df             = load_data()
model, encoder, features = load_model()
GENEROS        = sorted(encoder.classes_.tolist())

# ── CABECERA ──────────────────────────────────────────────────────────────────
st.title("🎵 Music Hit Predictor")
st.markdown(
    "Analiza el mercado musical con datos de **Last.fm** y predice si tu canción "
    "tiene potencial de convertirse en un hit."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🎵 Canciones en el dataset",  f"{len(df):,}")
col2.metric("🎤 Artistas únicos",           f"{df['artist'].nunique():,}")
col3.metric("🎧 Géneros únicos",            f"{df['tag'].nunique():,}")
col4.metric("🔥 Umbral de hit",             f"{int(P90_THRESHOLD / 1e6)}M reproducciones")

st.markdown("---")

# ── SECCIÓN 1: PREDICTOR ──────────────────────────────────────────────────────
st.header("🎯 ¿Será un hit?")
st.markdown("Introduce las características de tu canción y el modelo estimará su probabilidad de éxito.")

col_inputs, col_result = st.columns([1, 1], gap="large")

with col_inputs:
    st.subheader("Características de la canción")

    genero = st.selectbox(
        "🎧 Género musical",
        options=GENEROS,
        index=GENEROS.index("pop") if "pop" in GENEROS else 0,
        help="Selecciona el género más cercano al estilo de tu canción",
    )
    duracion = st.slider(
        "⏱️ Duración (minutos)",
        min_value=0.5, max_value=10.0, value=3.5, step=0.1,
        help="Duración de la canción en minutos",
    )
    oyentes = st.number_input(
        "👥 Oyentes únicos estimados",
        min_value=100, max_value=10_000_000, value=50_000, step=1_000,
        help="Número estimado de personas que escucharán la canción",
    )
    engagement = st.slider(
        "🔁 Reproducciones por oyente",
        min_value=1.0, max_value=30.0, value=5.0, step=0.5,
        help="Cuántas veces escucha cada oyente la canción de media",
    )

    predecir = st.button("🚀 Predecir potencial", use_container_width=True, type="primary")

with col_result:
    st.subheader("Resultado")

    if predecir:
        resultado = predecir_hit(model, encoder, features, duracion, genero, float(oyentes), engagement)

        prob  = resultado["probability"]
        label = resultado["label"]
        emoji = resultado["emoji"]

        st.markdown(f"### {emoji} {label}")

        col_a, col_b = st.columns(2)
        col_a.metric("Probabilidad de hit", f"{prob:.1f}%")
        col_b.metric("Duración", f"{duracion:.1f} min")

        st.progress(prob / 100)

        if prob >= 70:
            st.success(
                f"🚀 **Alto potencial.** Con {oyentes:,} oyentes y un engagement de "
                f"{engagement:.1f}x, la canción entraría en el top 10% del mercado."
            )
        elif prob >= 45:
            st.warning(
                "🟡 **Potencial medio.** Señales prometedoras. "
                "Aumentar oyentes o engagement podría llevarla al siguiente nivel."
            )
        else:
            st.error(
                "📉 **Potencial bajo** con los valores actuales. "
                "Prueba a aumentar los oyentes estimados o cambiar el género."
            )

        if resultado["is_short"]:
            st.info("⏱️ Canción corta (<2.5 min): compatible con formato TikTok y Reels.")

        with st.expander("🔍 Ver desglose de inputs del modelo"):
            st.write({
                "log_listeners (oyentes en escala log)"   : round(np.log1p(oyentes), 3),
                "duration_min"                            : duracion,
                "is_short_track (1=corta, 0=larga)"      : int(duracion < 2.5),
                "playcount_per_listener (engagement)"     : engagement,
                "plays estimados"                         : f"{oyentes * engagement:,.0f}",
            })
    else:
        st.info("👈 Completa los parámetros y pulsa **Predecir potencial** para ver el resultado.")

st.markdown("---")

# ── SECCIÓN 2: ANÁLISIS VISUAL ────────────────────────────────────────────────
st.header("📊 Análisis del mercado musical")
st.markdown("Explora los datos reales de Last.fm: canciones más populares, géneros y geografía.")

tab1, tab2, tab3 = st.tabs([
    "🏆 Top 15 canciones",
    "🎧 Géneros por popularidad",
    "🌍 Género × País",
])

with tab1:
    st.markdown("#### Las 15 canciones con más reproducciones en el dataset")
    st.pyplot(plot_top_tracks(df, n=15), use_container_width=True)

    top15 = (
        df[["name", "artist", "tag", "playcount"]]
        .dropna(subset=["playcount"])
        .sort_values("playcount", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    top15.index += 1
    top15["playcount"] = top15["playcount"].map(lambda x: f"{x/1e6:.1f}M")
    top15.columns = ["Canción", "Artista", "Género", "Reproducciones"]
    st.dataframe(top15, use_container_width=True)

with tab2:
    st.markdown("#### Reproducciones totales por género musical")
    n_generos = st.slider("Número de géneros a mostrar", 5, 20, 12, key="n_genres")
    st.pyplot(plot_genres(df, n=n_generos), use_container_width=True)
    pct = df["tag"].notna().mean() * 100
    st.caption(
        f"ℹ️ El {pct:.1f}% de los tracks tiene género asignado "
        f"({df['tag'].notna().sum():,} de {len(df):,}). "
        "Last.fm solo devuelve tags cuando los usuarios los han etiquetado."
    )

with tab3:
    st.markdown("#### Reproducciones medias por género y país de origen del dato")
    st.pyplot(plot_heatmap(df), use_container_width=True)
    real = df[~df["country"].isin(["UNKNOWN", "GLOBAL"])]["country"].notna().sum()
    st.caption(
        f"ℹ️ Solo {real:,} tracks tienen país real asignado ({real/len(df)*100:.1f}%). "
        "El resto proviene del endpoint global o de búsquedas por tag sin localización."
    )

st.markdown("---")
st.caption("Proyecto Final · Data Science Bootcamp · Datos: Last.fm API")
