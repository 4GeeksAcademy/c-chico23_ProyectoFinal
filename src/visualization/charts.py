"""
src/visualization/charts.py
Funciones de visualización para la app Streamlit.
Todas devuelven un objeto Figure de matplotlib (no hacen plt.show()).
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# Paleta consistente con el tema de la app
_PALETTE_BLUE   = "steelblue"
_PALETTE_ORANGE = "#E8774A"
_PALETTE_GREENS = "YlGn"

sns.set_theme(style="whitegrid", font_scale=1.0)


def plot_top_tracks(df: pd.DataFrame, n: int = 15) -> plt.Figure:
    """
    Barplot horizontal con las top N canciones por playcount.

    Parámetros:
        df — DataFrame con columnas 'name', 'artist', 'playcount'
        n  — número de canciones a mostrar (default 15)

    Devuelve:
        Figure de matplotlib
    """
    top = (
        df[["name", "artist", "playcount"]]
        .dropna(subset=["playcount"])
        .sort_values("playcount", ascending=False)
        .head(n)
        .copy()
    )
    # Etiqueta legible: "Canción — Artista"
    top["label"] = top["name"] + "  —  " + top["artist"]
    top = top.sort_values("playcount", ascending=True)  # ascendente para barh

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top["label"], top["playcount"] / 1e6, color=_PALETTE_BLUE, edgecolor="white")

    ax.set_xlabel("Reproducciones (millones)", fontsize=11)
    ax.set_title(f"🎵 Top {n} canciones por reproducciones", fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))

    # Etiqueta de valor al final de cada barra
    for bar, val in zip(bars, top["playcount"] / 1e6):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}M", va="center", fontsize=8.5, color="dimgray"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_genres(df: pd.DataFrame, n: int = 12) -> plt.Figure:
    """
    Barplot de géneros por reproducciones totales agregadas.

    Parámetros:
        df — DataFrame con columnas 'tag', 'playcount'
        n  — número de géneros a mostrar (default 12)

    Devuelve:
        Figure de matplotlib
    """
    df_tag = df.dropna(subset=["tag", "playcount"]).copy()

    if df_tag.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Sin datos de género disponibles", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("🎧 Géneros por popularidad")
        return fig

    stats = (
        df_tag.groupby("tag")["playcount"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    stats = stats.sort_values("playcount", ascending=True)

    colors = sns.color_palette("Blues_r", n_colors=len(stats))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(stats["tag"], stats["playcount"] / 1e9, color=colors, edgecolor="white")

    ax.set_xlabel("Reproducciones totales (miles de millones)", fontsize=11)
    ax.set_title(f"🎧 Top {n} géneros por popularidad", fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}B"))

    for bar, val in zip(bars, stats["playcount"] / 1e9):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}B", va="center", fontsize=8.5, color="dimgray"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Heatmap de reproducciones medias por Género × País.
    Solo incluye países reales (excluye GLOBAL y UNKNOWN).

    Parámetros:
        df — DataFrame con columnas 'tag', 'country', 'playcount'

    Devuelve:
        Figure de matplotlib
    """
    # Filtrar países reales y tracks con tag
    df_geo = df[
        df["country"].notna() &
        ~df["country"].isin(["UNKNOWN", "GLOBAL"]) &
        df["tag"].notna()
    ].copy()

    if df_geo.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Sin datos geográficos disponibles", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("📊 Género × País")
        return fig

    # Top 10 géneros y top 10 países por volumen
    top_tags = (
        df_geo.groupby("tag")["playcount"].sum()
        .sort_values(ascending=False).head(10).index.tolist()
    )
    top_countries = (
        df_geo.groupby("country")["playcount"].sum()
        .sort_values(ascending=False).head(10).index.tolist()
    )

    df_geo = df_geo[df_geo["tag"].isin(top_tags) & df_geo["country"].isin(top_countries)]

    pivot = (
        df_geo.groupby(["tag", "country"])["playcount"]
        .mean()
        .unstack(fill_value=0)
        / 1e6
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=_PALETTE_GREENS,
        annot=True,
        fmt=".1f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Reproducciones medias (M)", "shrink": 0.8},
    )
    ax.set_title("📊 Reproducciones medias por Género × País (millones)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("País", fontsize=11)
    ax.set_ylabel("Género", fontsize=11)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    return fig
