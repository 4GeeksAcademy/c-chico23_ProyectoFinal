"""
src/data/load_data.py
Carga y unión de los CSVs de LastFM.
Replica exactamente la lógica de carga del notebook (celdas 35, 56-58).
"""

import os
import pandas as pd

RAW_DIR       = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')


def load_backup_tracks(path: str = None) -> pd.DataFrame:
    """
    Carga backup_tracks.csv (generado por track.getInfo en el notebook).
    Columnas: name, artist, duration (ms), mbid, tag, streamable,
              listeners, playcount, published
    """
    if path is None:
        path = os.path.join(RAW_DIR, 'backup_tracks.csv')
    return pd.read_csv(path, low_memory=False)


def load_lastfm_dataset(path: str = None) -> pd.DataFrame:
    """
    Carga lastfm_dataset.csv (pipeline multi-endpoint).
    Columnas: name, artist, playcount, listeners, duration (s),
              mbid, country, genre_tag, rank_global, rank_by_country
    """
    if path is None:
        path = os.path.join(RAW_DIR, 'lastfm_dataset.csv')
    return pd.read_csv(path, low_memory=False)


def load_tags_dataset(path: str = None) -> pd.DataFrame:
    """Carga tags_dataset.csv (name, count, reach)."""
    if path is None:
        path = os.path.join(RAW_DIR, 'tags_dataset.csv')
    return pd.read_csv(path, low_memory=False)


def build_df_merged() -> pd.DataFrame:
    """
    Construye df_merged como lo hace el notebook (celdas 56-58).

    CORRECCIÓN aplicada: deduplicar lastfm_dataset por mbid antes del merge
    para evitar la explosión de 34k → 512k filas.
    El notebook original no lo hace — añadirlo aquí protege Streamlit.
    Prioridad de country: país real > GLOBAL > UNKNOWN.
    """
    df_backup = load_backup_tracks()
    df_lastfm = load_lastfm_dataset()

    # Deduplicar lastfm por mbid con prioridad de country
    df_lastfm['_priority'] = df_lastfm['country'].map(
        lambda x: 2 if x == 'UNKNOWN' else (1 if x == 'GLOBAL' else 0)
    )
    df_country = (
        df_lastfm.sort_values('_priority')
        .drop_duplicates(subset=['mbid'], keep='first')
        [['mbid', 'country']]
    )

    df_merged = df_backup.merge(df_country, on='mbid', how='left')
    return df_merged


def load_df_merged(path: str = None) -> pd.DataFrame:
    """
    Carga df_merged-data.csv si existe; si no, lo construye.
    """
    if path is None:
        path = os.path.join(PROCESSED_DIR, 'df_merged-data.csv')

    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False)

    print('df_merged-data.csv no encontrado — construyendo desde CSVs fuente...')
    return build_df_merged()
