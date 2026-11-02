import datetime as dt

import joblib as jb
import numpy as np
import pandas as pd
import streamlit as st


# ================================================================
# 0. CONFIGURATION DE LA PAGE
# ================================================================

st.set_page_config(
    page_title="CoinAfrique AI - Intelligence Prédictive",
    page_icon="🚗",
    layout="wide",
)


# ================================================================
# 1. CHARGEMENT DES ARTEFACTS (mis en cache pour ne charger qu'une fois)
# ================================================================

@st.cache_resource
def load_artifacts():
    encoders = jb.load("encoders.joblib")
    var_cat_vid = jb.load("var_cat_vid.joblib")
    scaler = jb.load("scaler.joblib")
    rf = jb.load("rf_model.joblib")
    return encoders, var_cat_vid, scaler, rf


try:
    encoders, var_cat_vid, scaler, rf = load_artifacts()
except Exception as e:
    st.error(
        "❌ Impossible de charger les fichiers du modèle "
        "(encoders.joblib, var_cat_vid.joblib, scaler.joblib, rf_model.joblib). "
        "Assure-toi qu'ils sont bien dans le même dossier que ce script.\n\n"
        f"Détail : {e}"
    )
    st.stop()

clasnames = var_cat_vid[3]

MARQUES = list(var_cat_vid[0])
TRANSMISSIONS = list(var_cat_vid[1])
QUARTIERS = list(var_cat_vid[2])


