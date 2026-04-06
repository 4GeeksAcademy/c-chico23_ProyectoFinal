"""
src/app.py
Aplicación Streamlit — Análisis del Mercado Musical con Last.fm

Ejecutar desde la raíz del proyecto:
    streamlit run src/app.py
"""

import os
import sys

import numpy as np
import streamlit as st

# Añadir src/ al path para imports relativos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.load_data           import load_df_clean
from models.predict           import predecir_hit, GENEROS_DISPONIBLES
from visualization.charts     import plot_top_tracks, plot_genres, plot_heatmap

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Music Hit Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS mínimo para ajuste visual ─────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .metric-label { font-size: 0.85rem !important; }
    hr { margin: 1.5rem 0; border: none; border-top: 1px solid #e0e0e0; }
</style>
""", unsafe_allow_html=True)


# ── Carga de datos (cacheada) ─────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando datos...")
def get_data():
    return load_df_clean()


df = get_data()


# ═══════════════════════════════════════════════════════════════════════════════
# CABECERA
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🎵 Music Hit Predictor")
st.markdown(
    "Analiza el mercado musical con datos de **Last.fm** y predice si tu canción "
    "tiene potencial de convertirse en un hit."
)

# Métricas rápidas del dataset
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎵 Total canciones", f"{len(df):,}")
col2.metric("🎤 Artistas únicos", f"{df['artist'].nunique():,}")
col3.metric("🎧 Géneros únicos", f"{df['tag'].nunique():,}")
col4.metric("🔥 Umbral de hit", f"{int(df['playcount'].quantile(0.90) / 1e6):.0f}M plays")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — PREDICTOR DE HIT
# ═══════════════════════════════════════════════════════════════════════════════
st.header("🎯 ¿Será un hit?")
st.markdown(
    "Introduce las características de tu canción y el modelo estimará su probabilidad de éxito."
)

col_inputs, col_result = st.columns([1, 1], gap="large")

with col_inputs:
    st.subheader("Características de la canción")

    genero = st.selectbox(
        "🎧 Género musical",
        options=GENEROS_DISPONIBLES,
        index=GENEROS_DISPONIBLES.index("pop") if "pop" in GENEROS_DISPONIBLES else 0,
        help="Selecciona el género más cercano al estilo de tu canción",
    )

    duracion = st.slider(
        "⏱️ Duración (minutos)",
        min_value=0.5,
        max_value=10.0,
        value=3.5,
        step=0.1,
        help="Duración de la canción en minutos",
    )

    oyentes = st.number_input(
        "👥 Oyentes únicos estimados",
        min_value=100,
        max_value=10_000_000,
        value=50_000,
        step=1_000,
        help="Estimación del número de personas que escucharán la canción",
    )

    engagement = st.slider(
        "🔁 Reproducciones por oyente (engagement)",
        min_value=1.0,
        max_value=30.0,
        value=5.0,
        step=0.5,
        help="Cuántas veces, de media, escucha cada oyente la canción",
    )

    predecir = st.button("🚀 Predecir potencial", use_container_width=True, type="primary")

with col_result:
    st.subheader("Resultado")

    if predecir:
        resultado = predecir_hit(
            duracion_min           = duracion,
            genero                 = genero,
            oyentes_estimados      = float(oyentes),
            playcount_per_listener = engagement,
        )

        prob  = resultado["probability"]
        label = resultado["label"]
        emoji = resultado["emoji"]

        # Resultado visual
        st.markdown(f"### {emoji} {label}")

        col_a, col_b = st.columns(2)
        col_a.metric("Probabilidad de hit", f"{prob:.1f}%")
        col_b.metric("Duración", f"{duracion:.1f} min")

        # Barra de progreso con color según resultado
        st.progress(prob / 100)

        # Contexto según clasificación
        if prob >= 70:
            st.success(
                f"🚀 **Alto potencial.** Con {oyentes:,} oyentes y engagement de "
                f"{engagement:.1f}x, la canción entraría en el top 10% del mercado."
            )
        elif prob >= 45:
            st.warning(
                "🟡 **Potencial medio.** Tiene señales prometedoras. "
                "Aumentar la base de oyentes o el engagement podría llevarla al siguiente nivel."
            )
        else:
            st.error(
                "📉 **Potencial bajo** con los valores actuales. "
                "Considera aumentar los oyentes estimados o ajustar el género."
            )

        if resultado["is_short"]:
            st.info("⏱️ Canción corta (<2.5 min): compatible con formato TikTok y Reels.")

        # Desglose técnico en expander
        with st.expander("Ver desglose de inputs del modelo"):
            est_plays = oyentes * engagement
            st.write({
                "log_listeners"          : round(np.log1p(oyentes), 3),
                "duration_min"           : duracion,
                "is_short_track"         : int(duracion < 2.5),
                "playcount_per_listener" : engagement,
                "plays_estimados"        : f"{est_plays:,.0f}",
            })
    else:
        st.info("👈 Completa los parámetros y pulsa **Predecir potencial** para ver el resultado.")


st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — ANÁLISIS VISUAL
# ═══════════════════════════════════════════════════════════════════════════════
st.header("📊 Análisis del mercado musical")
st.markdown("Explora los datos reales de Last.fm: canciones más populares, géneros y geografía.")

# ── Tab layout para los tres gráficos ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🏆 Top 15 canciones",
    "🎧 Géneros por popularidad",
    "🌍 Género × País",
])

with tab1:
    st.markdown("#### Las 15 canciones con más reproducciones en el dataset")
    fig_tracks = plot_top_tracks(df, n=15)
    st.pyplot(fig_tracks, use_container_width=True)

    # Tabla resumen debajo del gráfico
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
    fig_genres = plot_genres(df, n=n_generos)
    st.pyplot(fig_genres, use_container_width=True)

    pct_con_tag = df["tag"].notna().mean() * 100
    st.caption(
        f"ℹ️ El {pct_con_tag:.1f}% de los tracks tiene género asignado "
        f"({df['tag'].notna().sum():,} de {len(df):,}). "
        "Last.fm solo devuelve tags cuando los usuarios los han etiquetado."
    )

with tab3:
    st.markdown("#### Reproducciones medias por género y país de origen del dato")
    fig_heatmap = plot_heatmap(df)
    st.pyplot(fig_heatmap, use_container_width=True)

    real_countries = df[~df["country"].isin(["UNKNOWN", "GLOBAL"])]["country"].notna().sum()
    st.caption(
        f"ℹ️ Solo {real_countries:,} tracks tienen un país real asignado ({real_countries/len(df)*100:.1f}%). "
        "El resto proviene del endpoint global o de búsquedas por tag sin localización."
    )


st.markdown("---")
st.caption("Proyecto Final · Data Science Bootcamp · Datos: Last.fm API")
