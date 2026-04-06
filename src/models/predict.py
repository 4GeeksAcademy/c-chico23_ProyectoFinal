"""
src/models/predict.py
Carga modelo_hit.pkl, encoder_tag.pkl y features.pkl.
Expone predecir_hit() para la app Streamlit.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

# Asegurar que src/ está en el path para el import relativo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.process_data import build_input_row

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def _load_artifacts():
    """Carga los tres artefactos del modelo desde src/models/."""
    model    = joblib.load(os.path.join(_MODELS_DIR, "modelo_hit.pkl"))
    encoder  = joblib.load(os.path.join(_MODELS_DIR, "encoder_tag.pkl"))
    features = joblib.load(os.path.join(_MODELS_DIR, "features.pkl"))
    return model, encoder, features


# Carga única al importar el módulo
_model, _encoder, _features = _load_artifacts()

# Lista de géneros disponibles para el selectbox de la app
GENEROS_DISPONIBLES = sorted(_encoder.classes_.tolist())


def predecir_hit(
    duracion_min: float,
    genero: str,
    oyentes_estimados: float,
    playcount_per_listener: float = 5.0,
    artist_track_count: int = 1,
    track_share_of_artist: float = 1.0,
) -> dict:
    """
    Predice si una canción tiene potencial de hit.

    Parámetros:
        duracion_min            — duración en minutos (ej: 3.5)
        genero                  — género musical (tag de Last.fm, ej: 'pop')
        oyentes_estimados       — número estimado de oyentes únicos
        playcount_per_listener  — engagement estimado (default 5.0)
        artist_track_count      — tracks del artista en el dataset (default 1)
        track_share_of_artist   — peso del track en el catálogo del artista (default 1.0)

    Devuelve dict con:
        probability  — float 0-100
        label        — str clasificación
        emoji        — emoji del resultado
        is_short     — bool si la canción es corta (<2.5 min)
    """
    tag_enc = (
        _encoder.transform([genero])[0]
        if genero in _encoder.classes_
        else 0
    )

    input_df = build_input_row(
        log_listeners          = np.log1p(oyentes_estimados),
        duration_min           = duracion_min,
        tag_encoded            = tag_enc,
        artist_track_count     = artist_track_count,
        track_share_of_artist  = track_share_of_artist,
        playcount_per_listener = playcount_per_listener,
    )

    # reindex garantiza el orden exacto de columnas que espera el modelo
    X = input_df.reindex(columns=_features, fill_value=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob = _model.predict_proba(X)[0][1] * 100

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
