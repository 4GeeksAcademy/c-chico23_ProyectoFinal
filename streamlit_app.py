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

sns.set_theme(style="dark", font_scale=1.0)

# Paleta de colores del HTML
_BG      = "#0a0a0f"
_SURFACE = "#12121a"
_BORDER  = "#2a2a3a"
_ORANGE  = "#ff6b35"
_PURPLE  = "#7c3aed"
_CYAN    = "#06b6d4"
_GREEN   = "#10b981"
_YELLOW  = "#f59e0b"
_TEXT    = "#e8e8f0"
_MUTED   = "#7a7a9a"

def _style_ax(ax, fig):
    """Aplica fondo oscuro y estilo del HTML a cualquier figura."""
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_SURFACE)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.xaxis.label.set_color(_MUTED)
    ax.yaxis.label.set_color(_MUTED)
    ax.title.set_color(_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_top_tracks(df, n=15):
    top = (
        df[["name", "artist", "playcount"]]
        .dropna(subset=["playcount"])
        .sort_values("playcount", ascending=False)
        .head(n)
        .copy()
    )
    top["label"] = top["name"] + "  —  " + top["artist"]
    top = top.sort_values("playcount", ascending=True)

    # Top 3 en naranja, siguiente tercio en morado, resto en cyan
    n_bars = len(top)
    colors = []
    for i in range(n_bars):
        if i >= n_bars - 3:
            colors.append(_ORANGE)
        elif i >= n_bars - 8:
            colors.append(_PURPLE)
        else:
            colors.append(_CYAN)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top["label"], top["playcount"] / 1e6,
                   color=colors, edgecolor="none", height=0.65)

    ax.set_xlabel("Reproducciones (millones)", fontsize=10)
    ax.set_title(f"Top {n} canciones por reproducciones", fontsize=13,
                 fontweight="bold", pad=14, color=_TEXT)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))

    for bar, val in zip(bars, top["playcount"] / 1e6):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}M", va="center", fontsize=8, color=_MUTED)

    # Línea de referencia en 50M
    ax.axvline(50, color=_BORDER, linewidth=1, linestyle="--", alpha=0.6)

    _style_ax(ax, fig)
    plt.tight_layout()
    return fig


def plot_genres(df, n=12):
    df_tag = df.dropna(subset=["tag", "playcount"]).copy()

    if df_tag.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_SURFACE)
        ax.text(0.5, 0.5, "Sin datos de género disponibles",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color=_MUTED)
        return fig

    stats = (
        df_tag.groupby("tag")["playcount"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
        .sort_values("playcount", ascending=True)
    )

    # Gradiente cyan → morado según posición
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "html", [_PURPLE, _CYAN], N=len(stats)
    )
    colors = [mcolors.to_hex(cmap(i / max(len(stats) - 1, 1)))
              for i in range(len(stats))]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(stats["tag"], stats["playcount"] / 1e9,
                   color=colors, edgecolor="none", height=0.65)

    ax.set_xlabel("Reproducciones totales (miles de millones)", fontsize=10)
    ax.set_title(f"Top {n} géneros por popularidad", fontsize=13,
                 fontweight="bold", pad=14, color=_TEXT)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}B"))

    for bar, val in zip(bars, stats["playcount"] / 1e9):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}B", va="center", fontsize=8, color=_MUTED)

    _style_ax(ax, fig)
    plt.tight_layout()
    return fig


def plot_heatmap(df):
    df_geo = df[
        df["country"].notna() &
        ~df["country"].isin(["UNKNOWN", "GLOBAL"]) &
        df["tag"].notna()
    ].copy()

    if df_geo.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_SURFACE)
        ax.text(0.5, 0.5, "Sin datos geográficos disponibles",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color=_MUTED)
        return fig

    top_tags = (df_geo.groupby("tag")["playcount"].sum()
                .sort_values(ascending=False).head(8).index.tolist())
    top_countries = (df_geo.groupby("country")["playcount"].sum()
                     .sort_values(ascending=False).head(8).index.tolist())

    df_geo = df_geo[df_geo["tag"].isin(top_tags) & df_geo["country"].isin(top_countries)]
    pivot  = (df_geo.groupby(["tag", "country"])["playcount"]
              .mean().unstack(fill_value=0) / 1e6)

    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "html_heat", [_SURFACE, _PURPLE, _ORANGE], N=256
    )

    fig, ax = plt.subplots(figsize=(9, 5))          # más pequeño que antes
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    sns.heatmap(
        pivot, ax=ax,
        cmap=cmap,
        annot=True, fmt=".0f",
        linewidths=0.5, linecolor=_BG,
        annot_kws={"size": 8, "color": _TEXT},
        cbar_kws={"label": "Reprod. medias (M)", "shrink": 0.7},
    )

    ax.set_title("Reproducciones medias por Género × País (M)",
                 fontsize=12, fontweight="bold", pad=12, color=_TEXT)
    ax.set_xlabel("País", fontsize=9, color=_MUTED)
    ax.set_ylabel("Género", fontsize=9, color=_MUTED)
    ax.tick_params(colors=_MUTED, labelsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    # Colorbar styling
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(_MUTED)
    cbar.ax.tick_params(colors=_MUTED, labelsize=7)
    cbar.outline.set_edgecolor(_BORDER)

    fig.patch.set_facecolor(_BG)
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

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: ANÁLISIS ACÚSTICO — metadata_acoustic.csv
# Añadido al proyecto original sin modificar nada de lo anterior.
# Requiere: data/processed/metadata_acoustic.csv (generado con librosa)
# ═══════════════════════════════════════════════════════════════════════════════

import matplotlib.patches as mpatches
import re as _re

_META_CSV = os.path.join(_BASE, "data", "processed", "metadata_acoustic.csv")

@st.cache_data(show_spinner="Cargando metadata acústica...")
def load_acoustic():
    df = pd.read_csv(_META_CSV)
    # Quitar duplicado Chappell Roan
    df = df.drop_duplicates(subset=["bpm", "key", "chords"]).copy()
    # Nombre corto legible
    df["short"] = df["song"].apply(
        lambda s: _re.sub(r"\(.*?\)|Official.*|Lyrics.*|Audio.*", "", s)
        .strip(" -–").strip()
    )
    df["mode"] = df["key"].apply(lambda k: "Mayor" if "Major" in k else "Menor")
    return df

# Helpers de transposición a Do
_NOTE_NUM = {"C":0,"C#":1,"D":2,"D#":3,"E":4,"F":5,"F#":6,"G":7,"G#":8,"A":9,"A#":10,"B":11}
_NUM_NOTE = {v:k for k,v in _NOTE_NUM.items()}

CHORD_COLORS_ST = {
    "C":"#4A90D9","C#":"#357ABD","D":"#5BA85A","D#":"#4A9248",
    "E":"#E8A838","F":"#E07B3A","F#":"#C85A3A","G":"#9B59B6",
    "G#":"#8E44AD","A":"#E74C3C","A#":"#C0392B","B":"#1ABC9C",
    "Cm":"#2471A3","C#m":"#1A5276","Dm":"#1E8449","D#m":"#196F3D",
    "Em":"#B7770D","Fm":"#A04000","F#m":"#922B21","Gm":"#6C3483",
    "G#m":"#5B2C6F","Am":"#C0392B","A#m":"#922B21","Bm":"#117A65",
}
CHORD_EXPLAIN_ST = {
    "C":"Do Mayor","C#":"Do# Mayor","D":"Re Mayor","D#":"Re# Mayor",
    "E":"Mi Mayor","F":"Fa Mayor","F#":"Fa# Mayor","G":"Sol Mayor",
    "G#":"Sol# Mayor","A":"La Mayor","A#":"La# Mayor","B":"Si Mayor",
    "Cm":"Do menor","C#m":"Do# menor","Dm":"Re menor","D#m":"Re# menor",
    "Em":"Mi menor","Fm":"Fa menor","F#m":"Fa# menor","Gm":"Sol menor",
    "G#m":"Sol# menor","Am":"La menor","A#m":"La# menor","Bm":"Si menor",
}

def _parse_chords(s):
    return [c.strip() for c in s.replace("→","→").split("→") if c.strip()]

def _transpose_to_C(chord, key_root):
    is_minor = chord.endswith("m")
    root = chord.rstrip("m")
    if root not in _NOTE_NUM:
        return chord
    interval = (_NOTE_NUM[root] - _NOTE_NUM.get(key_root, 0)) % 12
    return _NUM_NOTE[interval] + ("m" if is_minor else "")

def plot_chords_all(df_ac):
    """Barras de colores con los acordes de cada canción."""
    n = len(df_ac)
    fig, axes = plt.subplots(n, 1, figsize=(11, n * 1.3))
    if n == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, df_ac.iterrows()):
        chords = _parse_chords(row["chords"])
        for i, chord in enumerate(chords):
            color = CHORD_COLORS_ST.get(chord, "#95A5A6")
            rect  = mpatches.FancyBboxPatch(
                (i, 0.1), 0.88, 0.8,
                boxstyle="round,pad=0.04",
                facecolor=color, edgecolor="white", linewidth=1.5
            )
            ax.add_patch(rect)
            ax.text(i+0.44, 0.5, chord,
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
        ax.set_xlim(-0.1, max(len(chords), 4))
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(
            f"{row['short']}  ·  {row['key']}  ·  {row['bpm']} BPM",
            fontsize=8.5, loc="left", pad=2
        )
    plt.tight_layout()
    return fig

def plot_bpm(df_ac):
    df_s = df_ac.sort_values("bpm")
    colors = ["#E74C3C" if b>=140 else "#E8A838" if b>=110 else "#3498DB"
              for b in df_s["bpm"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df_s["short"], df_s["bpm"], color=colors, edgecolor="white")
    ax.axvline(120, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    for bar, val in zip(bars, df_s["bpm"]):
        ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                str(val), va="center", fontsize=8.5)
    ax.set_xlabel("BPM"); ax.set_title("Tempo por canción", fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig

def plot_key_pie(df_ac):
    counts = df_ac["mode"].value_counts()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%",
           colors=["#3498DB","#E74C3C"], startangle=90,
           textprops={"fontsize":12})
    ax.set_title("Mayor vs Menor", fontweight="bold")
    plt.tight_layout()
    return fig

def plot_norm_heatmap(df_ac):
    def jaccard(a, b):
        sa, sb = set(a.split(" → ")), set(b.split(" → "))
        return len(sa & sb) / len(sa | sb) if (sa | sb) else 0

    norms, names = [], []
    for _, row in df_ac.iterrows():
        kr = row["key"].split()[0]
        ch = _parse_chords(row["chords"])
        norms.append(" → ".join([_transpose_to_C(c, kr) for c in ch]))
        names.append(row["short"][:18])

    n = len(names)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i][j] = jaccard(norms[i], norms[j])

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(sim, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim[i][j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if sim[i][j]>0.6 else "black")
    plt.colorbar(im, ax=ax, label="Similitud (0–1)", shrink=0.8)
    ax.set_title("Similitud de acordes normalizados a Do", fontweight="bold", pad=10)
    plt.tight_layout()
    return fig

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🎼 Análisis acústico — Top 15 tracks")
st.markdown(
    "Variables extraídas con **librosa** directamente de los archivos MP3: "
    "tempo (BPM), tonalidad (Key), energía y progresión de acordes."
)

try:
    df_ac = load_acoustic()

    tab_bpm, tab_key, tab_chords, tab_norm = st.tabs([
        "🥁 BPM", "🎵 Tonalidad", "🎸 Acordes", "🔄 Acordes en Do"
    ])

    with tab_bpm:
        st.markdown("#### Tempo (BPM) por canción")
        st.caption(
            "**BPM** = pulsaciones por minuto. "
            "🔴 ≥140 rápida · 🟡 110–139 media · 🔵 <110 lenta."
        )
        st.pyplot(plot_bpm(df_ac), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 Rápidas ≥140",  f"{(df_ac['bpm']>=140).sum()}")
        c2.metric("🟡 Medias 110–139", f"{((df_ac['bpm']>=110)&(df_ac['bpm']<140)).sum()}")
        c3.metric("🔵 Lentas <110",    f"{(df_ac['bpm']<110).sum()}")

    with tab_key:
        st.markdown("#### Tonalidad musical")
        st.caption(
            "Las tonalidades **mayores** suenan alegres; "
            "las **menores** suenan más oscuras o emotivas."
        )
        col_pie, col_list = st.columns([1, 2])
        with col_pie:
            st.pyplot(plot_key_pie(df_ac), use_container_width=True)
        with col_list:
            for _, row in df_ac.sort_values("key").iterrows():
                emoji = "☀️" if row["mode"] == "Mayor" else "🌙"
                st.markdown(f"{emoji} **{row['short']}** — {row['key']}")

    with tab_chords:
        st.markdown("#### Progresión de acordes")
        st.caption(
            "Colores **azules/verdes** = acordes mayores (alegres). "
            "Colores **rojos/oscuros** = acordes menores (emotivos)."
        )
        st.pyplot(plot_chords_all(df_ac), use_container_width=True)

    with tab_norm:
        st.markdown("#### Acordes normalizados a Do (C)")
        st.caption(
            "Transponemos todos los acordes como si la tonalidad fuese siempre Do. "
            "Así dos canciones en tonalidades distintas pueden compararse directamente."
        )
        # Tabla
        norm_rows = []
        for _, row in df_ac.iterrows():
            kr  = row["key"].split()[0]
            ch  = _parse_chords(row["chords"])
            norm = " → ".join([_transpose_to_C(c, kr) for c in ch])
            norm_rows.append({
                "Canción": row["short"],
                "Key":     row["key"],
                "Originales": row["chords"],
                "En Do":  norm,
            })
        st.dataframe(pd.DataFrame(norm_rows), use_container_width=True, hide_index=True)
        st.markdown("##### Similitud entre canciones")
        st.pyplot(plot_norm_heatmap(df_ac), use_container_width=True)

except FileNotFoundError:
    st.warning(
        "⚠️ No se encontró `data/processed/metadata_acoustic.csv`. "
        "Ejecuta primero `notebook/notebooks-mejoras/metadata.ipynb` para generarlo."
    )

st.markdown("---")
st.caption("Proyecto Final · Data Science Bootcamp · Datos: Last.fm API + librosa")
