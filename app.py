# import datetime as dt

# import joblib as jb
# import numpy as np
# import pandas as pd
# import streamlit as st


# # ================================================================
# # 0. CONFIGURATION DE LA PAGE
# # ================================================================

# st.set_page_config(
#     page_title="CoinAfrique AI - Intelligence Prédictive",
#     page_icon="🚗",
#     layout="wide",
# )


# # ================================================================
# # 1. CHARGEMENT DES ARTEFACTS (mis en cache pour ne charger qu'une fois)
# # ================================================================

# @st.cache_resource
# def load_artifacts():
#     encoders = jb.load("encoders.joblib")
#     var_cat_vid = jb.load("var_cat_vid.joblib")
#     scaler = jb.load("scaler.joblib")
#     rf = jb.load("rf_model.joblib")
#     return encoders, var_cat_vid, scaler, rf


# try:
#     encoders, var_cat_vid, scaler, rf = load_artifacts()
# except Exception as e:
#     st.error(
#         "❌ Impossible de charger les fichiers du modèle "
#         "(encoders.joblib, var_cat_vid.joblib, scaler.joblib, rf_model.joblib). "
#         "Assure-toi qu'ils sont bien dans le même dossier que ce script.\n\n"
#         f"Détail : {e}"
#     )
#     st.stop()

# clasnames = var_cat_vid[3]

# MARQUES = list(var_cat_vid[0])
# TRANSMISSIONS = list(var_cat_vid[1])
# QUARTIERS = list(var_cat_vid[2])


# # ================================================================
# # 2. HISTORIQUE DES PRÉDICTIONS (état de session Streamlit)
# # ================================================================

# HISTORY_COLUMNS = [
#     "Véhicule",
#     "Année",
#     "Prix",       # texte formaté pour l'affichage
#     "PrixNum",    # valeur numérique, utilisée pour les moyennes
#     "Quartier",
#     "État Prédit",
#     "Confiance",  # en %
#     "Horodatage",
# ]


# def _format_prix(prix):
#     try:
#         return f"{float(prix):,.0f} FCFA".replace(",", " ")
#     except (TypeError, ValueError):
#         return str(prix)


# def _add_to_history(marque, annee, prix, quartier, etat, confiance):
#     """Ajoute une prédiction à l'historique global du dashboard
#     (stocké dans st.session_state, propre à chaque session)."""

#     try:
#         annee_val = int(float(annee))
#     except (TypeError, ValueError):
#         annee_val = None

#     try:
#         prix_val = float(prix)
#     except (TypeError, ValueError):
#         prix_val = np.nan

#     new_row = {
#         "Véhicule": str(marque),
#         "Année": annee_val,
#         "Prix": _format_prix(prix),
#         "PrixNum": prix_val,
#         "Quartier": str(quartier),
#         "État Prédit": str(etat),
#         "Confiance": round(float(confiance) * 100, 1),
#         "Horodatage": dt.datetime.now(),
#     }

#     st.session_state.prediction_history = pd.concat(
#         [st.session_state.prediction_history, pd.DataFrame([new_row])],
#         ignore_index=True,
#     )


# def _seed_demo_history():
#     """Pré-remplit le dashboard avec quelques exemples au démarrage."""

#     now = dt.datetime.now()

#     demo_rows = [
#         ("Toyota Corolla", 2019, 5500000, "Almadies", "Très Bon", 96, 5),
#         ("Hyundai Tucson", 2016, 6800000, "Mermoz", "Bon", 89, 4),
#         ("Peugeot 308", 2012, 2300000, "Pikine", "Moyen", 72, 3),
#         ("Mercedes C-Class", 2021, 14000000, "Sacré-Cœur", "Excellent", 98, 2),
#         ("Toyota RAV4", 2020, 9200000, "Ouakam", "Très Bon", 94, 1),
#         ("Kia Sportage", 2018, 7100000, "Point E", "Bon", 87, 0),
#     ]

#     rows = []
#     for marque, annee, prix, quartier, etat, conf, jours in demo_rows:
#         rows.append({
#             "Véhicule": marque,
#             "Année": annee,
#             "Prix": _format_prix(prix),
#             "PrixNum": float(prix),
#             "Quartier": quartier,
#             "État Prédit": etat,
#             "Confiance": float(conf),
#             "Horodatage": now - dt.timedelta(days=jours),
#         })

#     return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


# # Initialisation de l'historique une seule fois par session
# if "prediction_history" not in st.session_state:
#     st.session_state.prediction_history = _seed_demo_history()


# # ================================================================
# # 3. PRÉDICTION D'UN VÉHICULE
# # ================================================================

# def predict_single(marque, transmission, quartier, annee, prix):

#     try:
#         marque_enc = encoders[0].transform([marque])[0]
#         trans_enc = encoders[1].transform([transmission])[0]
#         quartier_enc = encoders[2].transform([quartier])[0]
#     except ValueError as e:
#         raise ValueError(f"Valeur catégorielle inconnue du modèle : {e}")

#     # IMPORTANT : l'ordre doit correspondre exactement à celui utilisé
#     # lors de l'entraînement du scaler/modèle :
#     # [Marque, Année, Transmission, Prix, Quartier]
#     x_new = np.array([
#         marque_enc,
#         float(annee),
#         trans_enc,
#         float(prix),
#         quartier_enc,
#     ]).reshape(1, -1)

#     x_scaled = scaler.transform(x_new)
#     y_pred = rf.predict(x_scaled)[0]
#     probas = rf.predict_proba(x_scaled)[0]
#     confidence = float(np.max(probas))

#     return str(clasnames[y_pred]), confidence


# # ================================================================
# # 4. RECHERCHE DES COLONNES CSV
# # ================================================================

# def _find_column(df, candidates):
#     lower_map = {str(c).lower(): c for c in df.columns}
#     for cand in candidates:
#         if cand.lower() in lower_map:
#             return lower_map[cand.lower()]
#     return None


# # ================================================================
# # 5. PRÉDICTION CSV (batch)
# # ================================================================

# def predict_csv(df):
#     """Prend un DataFrame déjà lu, prédit chaque ligne, met à jour
#     l'historique de session, renvoie (df_resultat, message)."""

#     col_marque = _find_column(df, ["marque", "Marque"])
#     col_trans = _find_column(df, ["transmission", "Transmission"])
#     col_quartier = _find_column(df, ["quartier", "Quartier"])
#     col_annee = _find_column(df, ["annee", "année", "Annee", "Année"])
#     col_prix = _find_column(df, ["prix", "Prix"])

#     cols = list(df.columns)
#     col_marque = col_marque or (cols[0] if len(cols) > 0 else None)
#     col_trans = col_trans or (cols[1] if len(cols) > 1 else None)
#     col_quartier = col_quartier or (cols[2] if len(cols) > 2 else None)
#     col_annee = col_annee or (cols[3] if len(cols) > 3 else None)
#     col_prix = col_prix or (cols[4] if len(cols) > 4 else None)

#     if None in (col_marque, col_trans, col_quartier, col_annee, col_prix):
#         return None, (
#             "❌ Le CSV doit contenir 5 colonnes : "
#             "Marque, Transmission, Quartier, Annee, Prix."
#         )

#     predictions = []
#     confidences = []

#     for _, row in df.iterrows():
#         try:
#             pred, conf = predict_single(
#                 row[col_marque], row[col_trans], row[col_quartier],
#                 row[col_annee], row[col_prix],
#             )
#             _add_to_history(
#                 row[col_marque], row[col_annee], row[col_prix],
#                 row[col_quartier], pred, conf,
#             )
#         except Exception:
#             pred, conf = "Inconnu", 0.0

#         predictions.append(pred)
#         confidences.append(float(conf * 100))

#     df = df.copy()
#     df["Etat_Predit"] = predictions
#     df["Score_Confiance"] = [f"{x:.1f}%" for x in confidences]

#     nb_total = len(df)
#     nb_fiable = sum(x >= 80 for x in confidences)
#     score_moyen = np.mean(confidences) if confidences else 0

#     message = (
#         f"✅ {nb_total} lignes traitées | "
#         f"Score moyen : {score_moyen:.1f}% | "
#         f"Prédictions fiables : {nb_fiable}"
#     )

#     return df, message


# # ================================================================
# # 6. AFFICHAGE DES SCORES
# # ================================================================

# def score_badge(score):
#     if score >= 90:
#         return f"🟢 {score}%"
#     elif score >= 80:
#         return f"🔵 {score}%"
#     elif score >= 70:
#         return f"🟠 {score}%"
#     else:
#         return f"🔴 {score}%"


# # ================================================================
# # 7. CONSTRUCTION DYNAMIQUE DU TABLEAU DE BORD
# # ================================================================

# def _build_dashboard_table(history_df, max_rows=50):
#     if len(history_df) == 0:
#         return pd.DataFrame(
#             columns=["Véhicule", "Année", "Prix", "Quartier", "État Prédit", "Confiance"]
#         )

#     display_df = history_df.sort_values("Horodatage", ascending=False).head(max_rows).copy()
#     display_df["Confiance"] = display_df["Confiance"].apply(score_badge)

#     return display_df[["Véhicule", "Année", "Prix", "Quartier", "État Prédit", "Confiance"]]


# from datetime import date

# def _build_kpi_html(history_df):
#     total = len(history_df)

#     if total == 0:
#         avg_conf = 0.0
#         avg_prix_m = 0.0
#         today_count = 0
#     else:
#         avg_conf = history_df["Confiance"].mean()
#         prix_valides = history_df["PrixNum"].dropna()
#         avg_prix_m = prix_valides.mean() / 1_000_000 if len(prix_valides) else 0.0
        
#         history_df["Horodatage"] = pd.to_datetime(
#             history_df["Horodatage"],
#             errors="coerce"
#         )

#         today = date.today()  # <-- Définition de la variable manquante
        
#         today_count = int(
#             (history_df["Horodatage"].dt.date == today).sum()
#         )

#     conf_status = "✓ Performance validée" if avg_conf >= 80 else "⚠️ À surveiller"
     

#     return f"""
# <div class="kpi-container">
#     <div class="kpi-card kpi-blue">
#         <div class="kpi-icon">📊</div>
#         <div class="kpi-title">PRÉDICTIONS RÉALISÉES</div>
#         <div class="kpi-value">{total}</div>
#         <div class="kpi-positive">Total cumulé</div>
#     </div>
#     <div class="kpi-card kpi-green">
#         <div class="kpi-icon">🎯</div>
#         <div class="kpi-title">CONFIANCE MOYENNE</div>
#         <div class="kpi-value">{avg_conf:.1f} %</div>
#         <div class="kpi-positive">{conf_status}</div>
#     </div>
#     <div class="kpi-card kpi-orange">
#         <div class="kpi-icon">💰</div>
#         <div class="kpi-title">PRIX MOYEN ÉVALUÉ</div>
#         <div class="kpi-value">{avg_prix_m:.1f} M</div>
#         <div class="kpi-positive">FCFA</div>
#     </div>
#     <div class="kpi-card kpi-purple">
#         <div class="kpi-icon">⚡</div>
#         <div class="kpi-title">PRÉDICTIONS AUJOURD'HUI</div>
#         <div class="kpi-value">{today_count}</div>
#         <div class="kpi-positive">Depuis minuit</div>
#     </div>
# </div>
# """


# # ================================================================
# # 8. CSS PROFESSIONNEL
# # ================================================================

# CUSTOM_CSS = """
# <style>
# .stApp { background: #f4f7fb; }

# .dashboard-header {
#     background: #ffffff;
#     border-radius: 18px;
#     padding: 30px 35px;
#     margin-bottom: 25px;
#     border: 1px solid #dce3eb;
#     border-left: 7px solid #0b3d91;
#     box-shadow: 0 6px 20px rgba(16, 42, 67, 0.08);
# }
# .dashboard-title {
#     color: #0b2d5c;
#     font-size: 34px;
#     font-weight: 800;
#     line-height: 1.2;
#     letter-spacing: -0.5px;
# }
# .dashboard-subtitle {
#     color: #536579;
#     font-size: 16px;
#     margin-top: 10px;
#     font-weight: 500;
# }
# .kpi-container {
#     display: grid;
#     grid-template-columns: repeat(4, 1fr);
#     gap: 18px;
#     margin-bottom: 28px;
# }
# .kpi-card {
#     background: #ffffff;
#     border-radius: 16px;
#     padding: 23px;
#     min-height: 145px;
#     border: 1px solid #dce3eb;
#     box-shadow: 0 5px 18px rgba(16, 42, 67, 0.07);
# }
# .kpi-blue { border-top: 5px solid #1769aa; }
# .kpi-green { border-top: 5px solid #159447; }
# .kpi-orange { border-top: 5px solid #e28a00; }
# .kpi-purple { border-top: 5px solid #7048a8; }
# .kpi-icon { font-size: 28px; margin-bottom: 8px; }
# .kpi-title { color: #586779; font-size: 13px; font-weight: 700; letter-spacing: 0.4px; }
# .kpi-value { color: #102a43; font-size: 30px; font-weight: 800; margin-top: 5px; }
# .kpi-positive { color: #159447; font-size: 13px; font-weight: 700; margin-top: 7px; }
# .section-title {
#     background: #ffffff;
#     color: #102a43;
#     font-size: 21px;
#     font-weight: 800;
#     padding: 17px 20px;
#     border-radius: 13px;
#     border-left: 5px solid #1769aa;
#     margin-top: 25px;
#     margin-bottom: 17px;
#     box-shadow: 0 4px 14px rgba(16,42,67,0.05);
# }
# .info-card {
#     background: #ffffff;
#     border-radius: 14px;
#     padding: 22px;
#     border: 1px solid #dce3eb;
#     box-shadow: 0 4px 15px rgba(16,42,67,0.05);
#     min-height: 180px;
# }
# .info-card h3 { color: #102a43; font-size: 18px; font-weight: 800; margin-top: 0; }
# .info-card p { color: #536579; font-size: 14px; line-height: 1.7; }
# .legend-box {
#     background: #ffffff;
#     padding: 18px 20px;
#     border-radius: 13px;
#     border: 1px solid #dce3eb;
#     box-shadow: 0 4px 14px rgba(16,42,67,0.05);
#     margin-top: 15px;
# }
# .legend-title { color: #102a43; font-weight: 800; font-size: 15px; margin-bottom: 12px; }
# .legend-item { display: inline-block; color: #536579; font-size: 14px; margin-right: 25px; margin-bottom: 8px; }

# @media (max-width: 1000px) { .kpi-container { grid-template-columns: repeat(2, 1fr); } }
# @media (max-width: 600px) {
#     .kpi-container { grid-template-columns: 1fr; }
#     .dashboard-title { font-size: 26px; }
#     .dashboard-header { padding: 23px; }
# }
# </style>
# """

# st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# # ================================================================
# # 9. HEADER
# # ================================================================

# HEADER_HTML = """
# <div class="dashboard-header">
#     <div class="dashboard-title">🚗 Modèle de machine learning pour la prédiction de l'état des véhicules CoinAfrique</div>
#     <div class="dashboard-subtitle">
#         Tableau de bord intelligent pour la prédiction
#         automatique de l'état des véhicules
#     </div>
# </div>
# """

# st.markdown(HEADER_HTML, unsafe_allow_html=True)


# # ================================================================
# # 10. APPLICATION (onglets)
# # ================================================================

# tab_dashboard, tab_single, tab_csv = st.tabs(
#     ["📊 Tableau de bord", "🔍 Prédiction simple", "📁 Importation CSV"]
# )


# # ------------------------------------------------------------------
# # ONGLET 1 : DASHBOARD
# # ------------------------------------------------------------------

# with tab_dashboard:

#     st.markdown(_build_kpi_html(st.session_state.prediction_history), unsafe_allow_html=True)

#     col_a, col_b = st.columns(2)
#     with col_a:
#         if st.button("🔄 Actualiser le tableau de bord", use_container_width=True):
#             st.rerun()
#     with col_b:
#         if st.button("🗑️ Réinitialiser l'historique", use_container_width=True):
#             st.session_state.prediction_history = pd.DataFrame(columns=HISTORY_COLUMNS)
#             st.success("🗑️ Historique réinitialisé. Le tableau de bord repart à zéro.")
#             st.rerun()

#     st.markdown('<div class="section-title">🎯 Performance et qualité des prédictions</div>', unsafe_allow_html=True)

#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown(
#             """
#             <div class="info-card">
#                 <h3>🤖 Modèle de Machine Learning</h3>
#                 <p>Le système utilise un modèle <b>Random Forest</b> pour prédire
#                 automatiquement l'état probable d'un véhicule.</p>
#                 <p><b>Variables utilisées :</b><br>
#                 Marque • Transmission • Quartier • Année • Prix</p>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )
#     with col2:
#         st.markdown(
#             """
#             <div class="info-card">
#                 <h3>🛡️ Niveau de fiabilité</h3>
#                 <p>Le score de confiance représente le niveau de certitude
#                 du modèle pour chaque prédiction.</p>
#                 <p><b>Seuil recommandé :</b> 80 %</p>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

#     st.markdown(
#         """
#         <div class="legend-box">
#             <div class="legend-title">📌 Interprétation du score de confiance</div>
#             <span class="legend-item">🟢 <b>≥ 90 %</b> — Très fiable</span>
#             <span class="legend-item">🔵 <b>80–89 %</b> — Fiable</span>
#             <span class="legend-item">🟠 <b>70–79 %</b> — À vérifier</span>
#             <span class="legend-item">🔴 <b>&lt; 70 %</b> — Faible confiance</span>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     st.markdown('<div class="section-title">🚗 Historique des dernières prédictions</div>', unsafe_allow_html=True)

#     st.dataframe(
#         _build_dashboard_table(st.session_state.prediction_history),
#         use_container_width=True,
#         hide_index=True,
#     )

#     st.markdown('<div class="section-title">📈 Synthèse décisionnelle</div>', unsafe_allow_html=True)

#     st.markdown(
#         """
#         <div class="info-card">
#             <h3>💡 Interprétation du tableau de bord</h3>
#             <p>L'application permet d'automatiser l'évaluation de l'état des
#             véhicules à partir de leurs caractéristiques. Les chiffres ci-dessus
#             reflètent les prédictions réellement effectuées via les onglets
#             "Prédiction simple" et "Importation CSV".</p>
#             <p>Un score de confiance élevé indique que le modèle est fortement
#             certain de sa prédiction. Les résultats présentant un score faible
#             doivent être vérifiés avant toute décision.</p>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )


# # ------------------------------------------------------------------
# # ONGLET 2 : PRÉDICTION SIMPLE
# # ------------------------------------------------------------------

# with tab_single:

#     st.markdown("## 🔍 Évaluation individuelle d'un véhicule")
#     st.write("Renseignez les caractéristiques du véhicule pour obtenir une prédiction automatique.")

#     col_left, col_right = st.columns([2, 1])

#     with col_left:
#         marque = st.selectbox("🚗 Marque", MARQUES, key="marque_in")

#         c1, c2 = st.columns(2)
#         with c1:
#             transmission = st.selectbox("⚙️ Transmission", TRANSMISSIONS, key="trans_in")
#         with c2:
#             quartier = st.selectbox("📍 Quartier", QUARTIERS, key="quartier_in")

#         c3, c4 = st.columns(2)
#         with c3:
#             annee = st.number_input("📅 Année", value=2018, step=1, format="%d", key="annee_in")
#         with c4:
#             prix = st.number_input("💰 Prix (FCFA)", value=4500000, step=100000, key="prix_in")

#         launch = st.button("🚀 Lancer la prédiction", type="primary", use_container_width=True)

#     with col_right:
#         result_placeholder = st.empty()
#         confidence_placeholder = st.empty()

#         if launch:
#             if not marque or not transmission or not quartier:
#                 st.warning("⚠️ Veuillez sélectionner toutes les options.")
#             else:
#                 try:
#                     prediction, confidence = predict_single(marque, transmission, quartier, annee, prix)
#                     _add_to_history(marque, annee, prix, quartier, prediction, confidence)
#                     result_placeholder.success(f"🎯 État évalué : **{prediction}**")
#                     confidence_placeholder.info(f"📊 Score de confiance : **{confidence * 100:.1f}%**")
#                 except ValueError as e:
#                     st.error(f"❌ Erreur : {e}")


# # ------------------------------------------------------------------
# # ONGLET 3 : IMPORTATION CSV
# # ------------------------------------------------------------------

# with tab_csv:

#     st.markdown("## 📁 Analyse prédictive en masse")
#     st.write("Importez un fichier CSV contenant les colonnes :")
#     st.markdown("**Marque • Transmission • Quartier • Annee • Prix**")

#     uploaded_file = st.file_uploader("📤 Importer votre fichier CSV", type=["csv"])

#     if st.button("🚀 Traiter le fichier", type="primary"):
#         if uploaded_file is None:
#             st.warning("⚠️ Veuillez importer un fichier CSV.")
#         else:
#             try:
#                 df_in = pd.read_csv(uploaded_file)
#             except Exception as e:
#                 st.error(f"❌ Impossible de lire le CSV : {e}")
#                 df_in = None

#             if df_in is not None:
#                 df_out, message = predict_csv(df_in)

#                 if df_out is None:
#                     st.error(message)
#                 else:
#                     st.success(message)
#                     st.dataframe(df_out, use_container_width=True, hide_index=True)

#                     csv_bytes = df_out.to_csv(index=False).encode("utf-8")
#                     st.download_button(
#                         "📥 Télécharger les résultats",
#                         data=csv_bytes,
#                         file_name="predictions_resultats.csv",
#                         mime="text/csv",
#                     )
import base64
import datetime as dt
import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor
from html import escape
from urllib.parse import quote

import joblib as jb
import numpy as np
import pandas as pd
import requests
import streamlit as st


# ================================================================
# 0. CONFIGURATION DE LA PAGE
# ================================================================

st.set_page_config(
    page_title="CoinAfrique AI - Intelligence Prédictive",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

RED = "#e10600"
DARK_RED = "#8a0000"


# ================================================================
# 1. CHARGEMENT DES ARTEFACTS
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


# ================================================================
# 2. UTILITAIRES VISUELS (illustrations vectorielles + photos perso)
# ================================================================

def _h(s):
    """Aplatit un bloc HTML pour éviter que Markdown le prenne pour du code."""
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


def _data_uri_svg(svg):
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode()


def _data_uri_file(path):
    ext = os.path.splitext(path)[1].lower().strip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "jpeg")
    with open(path, "rb") as f:
        return f"data:image/{mime};base64," + base64.b64encode(f.read()).decode()


# ---------- VRAIES PHOTOS DE VOITURES (Wikipédia / Wikimedia Commons) ----------

# Modèle emblématique par marque : la page du modèle contient une vraie photo de voiture
# (la page de la marque contient souvent un logo).
MODEL_HINTS = {
    "toyota": "Toyota GR Supra", "hyundai": "Hyundai Ioniq 5", "kia": "Kia EV6",
    "peugeot": "Peugeot 508", "renault": "Renault Mégane E-Tech", "citroen": "Citroën C5 X",
    "citroën": "Citroën C5 X", "mercedes": "Mercedes-AMG GT", "bmw": "BMW M4",
    "audi": "Audi RS e-tron GT", "volkswagen": "Volkswagen Golf R", "vw": "Volkswagen Golf R",
    "ford": "Ford Mustang (seventh generation)", "nissan": "Nissan GT-R", "honda": "Honda Civic Type R",
    "mazda": "Mazda MX-5", "suzuki": "Suzuki Jimny", "mitsubishi": "Mitsubishi Outlander",
    "lexus": "Lexus LC", "land rover": "Range Rover (L460)", "range rover": "Range Rover (L460)",
    "jeep": "Jeep Wrangler (JL)", "fiat": "Fiat 500e", "opel": "Opel Astra",
    "dacia": "Dacia Duster", "isuzu": "Isuzu D-Max", "subaru": "Subaru WRX",
    "volvo": "Volvo EX90", "porsche": "Porsche 911 (992)", "tesla": "Tesla Model S",
    "skoda": "Škoda Enyaq", "škoda": "Škoda Enyaq", "seat": "SEAT León",
    "chevrolet": "Chevrolet Corvette (C8)", "infiniti": "Infiniti QX80", "mini": "Mini Cooper",
    "jaguar": "Jaguar F-Type", "chery": "Chery Tiggo 8", "mg": "MG Cyberster",
    "ssangyong": "SsangYong Torres", "daihatsu": "Daihatsu Rocky", "lada": "Lada Niva",
    "dodge": "Dodge Challenger", "cadillac": "Cadillac Lyriq", "gmc": "GMC Hummer EV",
    "great wall": "Haval H6", "geely": "Geely Monjaro", "byd": "BYD Seal",
    "ferrari": "Ferrari 296 GTB", "lamborghini": "Lamborghini Revuelto",
    "bentley": "Bentley Continental GT", "rolls-royce": "Rolls-Royce Spectre",
    "rolls royce": "Rolls-Royce Spectre", "mclaren": "McLaren 750S",
    "aston martin": "Aston Martin DB12", "maserati": "Maserati MC20",
    "lotus": "Lotus Emira", "alfa romeo": "Alfa Romeo Giulia", "genesis": "Genesis G80",
    "polestar": "Polestar 2", "lucid": "Lucid Air",
}

# Vitrine : modèles d'exception récents (hero + bandeau dédié)
SHOWCASE = [
    "Lamborghini Revuelto", "Ferrari 296 GTB", "Porsche 911 (992)", "McLaren 750S",
    "Rolls-Royce Spectre", "Bentley Continental GT", "Mercedes-AMG GT", "Aston Martin DB12",
    "Maserati MC20", "BMW M4", "Audi RS e-tron GT", "Tesla Model S", "Lucid Air",
    "Lotus Emira", "Range Rover (L460)", "Mercedes-Benz G-Class", "Lamborghini Urus",
    "Chevrolet Corvette (C8)",
]


_WIKI_HEADERS = {"User-Agent": "CoinAfriqueAI/1.0 (Streamlit demo app)"}


def _wiki_thumb(params):
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php", params=params,
                         headers=_WIKI_HEADERS, timeout=6)
        if r.status_code != 200:
            return None
        pages = r.json().get("query", {}).get("pages", {})
        for p in pages.values():
            src = (p.get("thumbnail") or {}).get("source")
            if src and ".svg" not in src.lower():
                return src
    except Exception:
        return None
    return None


def _find_car_photo(name):
    name = str(name).strip()
    low = name.lower()
    candidates = []
    for key, model in MODEL_HINTS.items():
        if low == key or low.startswith(key + " "):
            # nom complet (ex. "Toyota Corolla") : on essaie d'abord le nom exact
            if low != key:
                candidates.append(name)
            candidates.append(model)
            break
    if not candidates:
        candidates.append(name)

    for c in candidates:
        url = _wiki_thumb({
            "action": "query", "format": "json", "redirects": 1, "titles": c,
            "prop": "pageimages", "piprop": "thumbnail", "pithumbsize": 800,
        })
        if url:
            return url
    return _wiki_thumb({
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"{candidates[-1]} car", "gsrlimit": 1,
        "prop": "pageimages", "piprop": "thumbnail", "pithumbsize": 800,
    })


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_car_photos(names):
    """names : tuple de noms. Renvoie {nom: url_photo_ou_None}. Mis en cache 24 h."""
    names = list(dict.fromkeys(str(n) for n in names))
    with ThreadPoolExecutor(max_workers=8) as ex:
        urls = list(ex.map(_find_car_photo, names))
    return dict(zip(names, urls))


def _photo_or_art(url, idx=0):
    """Vraie photo de voiture ; cadre neutre si la photo est introuvable."""
    if url:
        return f'<img class="mq-photo" src="{escape(url)}" alt="" referrerpolicy="no-referrer"/>'
    return '<div class="mq-ph">🚗</div>'





# ================================================================
# 3. HISTORIQUE DES PRÉDICTIONS
# ================================================================

HISTORY_COLUMNS = [
    "Véhicule", "Année", "Prix", "PrixNum", "Quartier",
    "État Prédit", "Confiance", "Horodatage",
]


def _format_prix(prix):
    try:
        return f"{float(prix):,.0f} FCFA".replace(",", " ")
    except (TypeError, ValueError):
        return str(prix)


def _add_to_history(marque, annee, prix, quartier, etat, confiance):
    try:
        annee_val = int(float(annee))
    except (TypeError, ValueError):
        annee_val = None
    try:
        prix_val = float(prix)
    except (TypeError, ValueError):
        prix_val = np.nan

    new_row = {
        "Véhicule": str(marque),
        "Année": annee_val,
        "Prix": _format_prix(prix),
        "PrixNum": prix_val,
        "Quartier": str(quartier),
        "État Prédit": str(etat),
        "Confiance": round(float(confiance) * 100, 1),
        "Horodatage": dt.datetime.now(),
    }
    st.session_state.prediction_history = pd.concat(
        [st.session_state.prediction_history, pd.DataFrame([new_row])],
        ignore_index=True,
    )


def _seed_demo_history():
    now = dt.datetime.now()
    demo_rows = [
        ("Toyota Corolla", 2019, 5500000, "Almadies", "Très Bon", 96, 5),
        ("Hyundai Tucson", 2016, 6800000, "Mermoz", "Bon", 89, 4),
        ("Peugeot 308", 2012, 2300000, "Pikine", "Moyen", 72, 3),
        ("Mercedes C-Class", 2021, 14000000, "Sacré-Cœur", "Excellent", 98, 2),
        ("Toyota RAV4", 2020, 9200000, "Ouakam", "Très Bon", 94, 1),
        ("Kia Sportage", 2018, 7100000, "Point E", "Bon", 87, 0),
    ]
    rows = []
    for marque, annee, prix, quartier, etat, conf, jours in demo_rows:
        rows.append({
            "Véhicule": marque, "Année": annee, "Prix": _format_prix(prix),
            "PrixNum": float(prix), "Quartier": quartier, "État Prédit": etat,
            "Confiance": float(conf), "Horodatage": now - dt.timedelta(days=jours),
        })
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = _seed_demo_history()


# ================================================================
# 4. PRÉDICTION
# ================================================================

def predict_single(marque, transmission, quartier, annee, prix):
    try:
        marque_enc = encoders[0].transform([marque])[0]
        trans_enc = encoders[1].transform([transmission])[0]
        quartier_enc = encoders[2].transform([quartier])[0]
    except ValueError as e:
        raise ValueError(f"Valeur catégorielle inconnue du modèle : {e}")

    # Ordre d'entraînement : [Marque, Année, Transmission, Prix, Quartier]
    x_new = np.array([
        marque_enc, float(annee), trans_enc, float(prix), quartier_enc,
    ]).reshape(1, -1)

    x_scaled = scaler.transform(x_new)
    y_pred = rf.predict(x_scaled)[0]
    probas = rf.predict_proba(x_scaled)[0]
    return str(clasnames[y_pred]), float(np.max(probas))


def _find_column(df, candidates):
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def predict_csv(df):
    col_marque = _find_column(df, ["marque"])
    col_trans = _find_column(df, ["transmission"])
    col_quartier = _find_column(df, ["quartier"])
    col_annee = _find_column(df, ["annee", "année"])
    col_prix = _find_column(df, ["prix"])

    cols = list(df.columns)
    col_marque = col_marque or (cols[0] if len(cols) > 0 else None)
    col_trans = col_trans or (cols[1] if len(cols) > 1 else None)
    col_quartier = col_quartier or (cols[2] if len(cols) > 2 else None)
    col_annee = col_annee or (cols[3] if len(cols) > 3 else None)
    col_prix = col_prix or (cols[4] if len(cols) > 4 else None)

    if None in (col_marque, col_trans, col_quartier, col_annee, col_prix):
        return None, "❌ Le CSV doit contenir 5 colonnes : Marque, Transmission, Quartier, Annee, Prix."

    predictions, confidences = [], []
    for _, row in df.iterrows():
        try:
            pred, conf = predict_single(
                row[col_marque], row[col_trans], row[col_quartier],
                row[col_annee], row[col_prix],
            )
            _add_to_history(row[col_marque], row[col_annee], row[col_prix],
                            row[col_quartier], pred, conf)
        except Exception:
            pred, conf = "Inconnu", 0.0
        predictions.append(pred)
        confidences.append(float(conf * 100))

    df = df.copy()
    df["Etat_Predit"] = predictions
    df["Score_Confiance"] = [f"{x:.1f}%" for x in confidences]

    nb_fiable = sum(x >= 80 for x in confidences)
    score_moyen = np.mean(confidences) if confidences else 0
    message = (
        f"✅ {len(df)} lignes traitées | Score moyen : {score_moyen:.1f}% | "
        f"Prédictions fiables : {nb_fiable}"
    )
    return df, message


# ================================================================
# 5. COMPOSANTS HTML
# ================================================================

def score_badge(score):
    if score >= 90:
        return "🟢", "b-green"
    if score >= 80:
        return "🔵", "b-blue"
    if score >= 70:
        return "🟠", "b-orange"
    return "🔴", "b-red"


def _etat_class(etat):
    e = etat.lower()
    if "excellent" in e or "très" in e or "tres" in e:
        return "s-top"
    if "bon" in e:
        return "s-good"
    if "moyen" in e:
        return "s-mid"
    return "s-low"


def build_hero():
    hero_img = None
    for ext in ("jpg", "jpeg", "png", "webp"):
        p = os.path.join("assets", f"hero.{ext}")
        if os.path.exists(p):
            hero_img = f'<img class="hero-img" src="{_data_uri_file(p)}" alt="Voitures"/>'
            break

    if hero_img is None:
        names = tuple(SHOWCASE[:5])
        photos = fetch_car_photos(names)
        urls = [u for u in photos.values() if u]
        if len(urls) >= 3:
            imgs = "".join(
                f'<img src="{escape(u)}" alt="" referrerpolicy="no-referrer"/>' for u in urls
            )
            hero_img = f'<div class="hero-collage">{imgs}</div>'
        else:
            hero_img = '<div class="hero-fallback"></div>'

    return _h(f"""
    <div class="hero">
      {hero_img}
      <div class="hero-overlay"></div>
      <div class="hero-content">
        <div class="hero-tag">⚡ INTELLIGENCE PRÉDICTIVE AUTOMOBILE</div>
        <h1 class="hero-title">COIN<span>AFRIQUE</span> AI</h1>
        <p class="hero-sub">Évaluez l'état de n'importe quel véhicule en quelques secondes
        grâce à un modèle Random Forest entraîné sur le marché réel.</p>
        <div class="hero-chips">
          <span>🚗 Prédiction instantanée</span>
          <span>📁 Analyse CSV en masse</span>
          <span>🎯 Score de confiance</span>
        </div>
      </div>
    </div>
    """)


def build_marquee(history_df):
    df = history_df.sort_values("Horodatage", ascending=False).head(12)
    if len(df) == 0:
        return _h('<div class="empty">Aucun véhicule analysé pour le moment. Lancez une prédiction !</div>')

    photos = fetch_car_photos(tuple(df["Véhicule"].astype(str)))
    cards = []
    for i, (_, r) in enumerate(df.iterrows()):
        visual = _photo_or_art(photos.get(str(r["Véhicule"])), i)
        icon, _ = score_badge(r["Confiance"])
        annee = "" if pd.isna(r["Année"]) else int(r["Année"])
        cards.append(f"""
        <div class="mq-card">
          {visual}
          <div class="mq-name">{escape(str(r["Véhicule"]))}</div>
          <div class="mq-meta">{annee} • {escape(str(r["Quartier"]))}</div>
          <div class="mq-price">{escape(str(r["Prix"]))}</div>
          <div class="mq-foot">
            <span class="chip {_etat_class(str(r["État Prédit"]))}">{escape(str(r["État Prédit"]))}</span>
            <span class="mq-conf">{icon} {r["Confiance"]}%</span>
          </div>
        </div>""")

    content = "".join(cards)
    return _h(f'<div class="marquee"><div class="mq-track">{content}{content}</div></div>')


def build_brand_marquee(max_brands=30):
    brands = [str(m) for m in MARQUES][:max_brands]
    if not brands:
        return ""
    photos = fetch_car_photos(tuple(brands))
    cards = []
    for i, b in enumerate(brands):
        cards.append(f"""
        <div class="mq-card mq-brand">
          {_photo_or_art(photos.get(b), i)}
          <div class="mq-name">{escape(b)}</div>
        </div>""")
    content = "".join(cards)
    return _h(
        f'<div class="marquee"><div class="mq-track mq-reverse">{content}{content}</div></div>'
    )


def build_showcase_marquee():
    photos = fetch_car_photos(tuple(SHOWCASE))
    cards = []
    for title in SHOWCASE:
        url = photos.get(title)
        if not url:
            continue
        label = re.sub(r"\s*\(.*?\)", "", title)
        cards.append(f"""
        <div class="mq-card mq-show">
          {_photo_or_art(url)}
          <div class="mq-name">{escape(label)}</div>
          <div class="mq-meta">Modèle d'exception</div>
        </div>""")
    if not cards:
        return ""
    content = "".join(cards)
    return _h(f'<div class="marquee"><div class="mq-track">{content}{content}</div></div>')


def build_kpis(history_df):
    df = history_df.copy()
    total = len(df)
    if total == 0:
        avg_conf, avg_prix_m, today_count = 0.0, 0.0, 0
    else:
        avg_conf = df["Confiance"].mean()
        pv = df["PrixNum"].dropna()
        avg_prix_m = pv.mean() / 1_000_000 if len(pv) else 0.0
        df["Horodatage"] = pd.to_datetime(df["Horodatage"], errors="coerce")
        today_count = int((df["Horodatage"].dt.date == dt.date.today()).sum())

    status = "✓ Performance validée" if avg_conf >= 80 else "⚠️ À surveiller"
    return _h(f"""
    <div class="kpi-container">
      <div class="kpi-card"><div class="kpi-icon">📊</div>
        <div class="kpi-title">PRÉDICTIONS RÉALISÉES</div>
        <div class="kpi-value">{total}</div><div class="kpi-sub">Total cumulé</div></div>
      <div class="kpi-card"><div class="kpi-icon">🎯</div>
        <div class="kpi-title">CONFIANCE MOYENNE</div>
        <div class="kpi-value">{avg_conf:.1f}%</div><div class="kpi-sub">{status}</div></div>
      <div class="kpi-card"><div class="kpi-icon">💰</div>
        <div class="kpi-title">PRIX MOYEN ÉVALUÉ</div>
        <div class="kpi-value">{avg_prix_m:.1f} M</div><div class="kpi-sub">FCFA</div></div>
      <div class="kpi-card"><div class="kpi-icon">⚡</div>
        <div class="kpi-title">PRÉDICTIONS AUJOURD'HUI</div>
        <div class="kpi-value">{today_count}</div><div class="kpi-sub">Depuis minuit</div></div>
    </div>
    """)


def _bars(title, items):
    """items: liste de (label, valeur, texte_affiché)."""
    if not items:
        return f'<div class="panel"><div class="panel-title">{title}</div><div class="empty">Pas de données</div></div>'
    vmax = max(v for _, v, _ in items) or 1
    rows = ""
    for label, v, txt in items:
        w = max(4, int(v / vmax * 100))
        rows += (
            f'<div class="bar-row"><div class="bar-label">{escape(str(label))}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{w}%"></div></div>'
            f'<div class="bar-val">{txt}</div></div>'
        )
    return _h(f'<div class="panel"><div class="panel-title">{title}</div>{rows}</div>')


def build_charts(history_df):
    if len(history_df) == 0:
        return _bars("📊 Répartition des états prédits", []), _bars("🎯 Niveaux de confiance", [])

    counts = history_df["État Prédit"].value_counts()
    etats = [(k, int(v), str(int(v))) for k, v in counts.items()]

    conf = history_df["Confiance"].astype(float)
    buckets = [
        ("🟢 ≥ 90 %", int((conf >= 90).sum())),
        ("🔵 80–89 %", int(((conf >= 80) & (conf < 90)).sum())),
        ("🟠 70–79 %", int(((conf >= 70) & (conf < 80)).sum())),
        ("🔴 < 70 %", int((conf < 70).sum())),
    ]
    confs = [(k, v, str(v)) for k, v in buckets]
    return _bars("📊 Répartition des états prédits", etats), _bars("🎯 Niveaux de confiance", confs)


def build_table(history_df, max_rows=50):
    if len(history_df) == 0:
        return '<div class="empty">Historique vide.</div>'
    df = history_df.sort_values("Horodatage", ascending=False).head(max_rows)
    rows = ""
    for _, r in df.iterrows():
        icon, bcls = score_badge(r["Confiance"])
        annee = "" if pd.isna(r["Année"]) else int(r["Année"])
        rows += (
            "<tr>"
            f'<td class="t-strong">🚗 {escape(str(r["Véhicule"]))}</td>'
            f"<td>{annee}</td><td>{escape(str(r['Prix']))}</td>"
            f"<td>{escape(str(r['Quartier']))}</td>"
            f'<td><span class="chip {_etat_class(str(r["État Prédit"]))}">{escape(str(r["État Prédit"]))}</span></td>'
            f'<td><span class="badge {bcls}">{icon} {r["Confiance"]}%</span></td>'
            "</tr>"
        )
    return _h(f"""
    <div class="table-wrap"><table class="dt">
      <thead><tr><th>Véhicule</th><th>Année</th><th>Prix</th><th>Quartier</th>
      <th>État prédit</th><th>Confiance</th></tr></thead>
      <tbody>{rows}</tbody></table></div>
    """)


# ================================================================
# 6. CSS ROUGE & NOIR (forcé, quel que soit le mode du navigateur)
# ================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');

:root { --red:#e10600; --red2:#ff2d20; --dred:#8a0000; --bg:#070707; --card:#141414; --card2:#1c1c1c; --line:#2c2c2c; --txt:#ffffff; --mut:#b9b9b9; }

html, body, .stApp, [class*="css"] { font-family:'Inter',sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 80% -10%, #3a0000 0%, transparent 60%), var(--bg) !important; color: var(--txt); }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.2rem; max-width: 1400px; }

.stApp p, .stApp label, .stApp span, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp div[data-testid="stMarkdownContainer"] { color: var(--txt); }
.stApp label p { color:#fff !important; font-weight:700 !important; }

/* ---------- HERO ---------- */
.hero { position:relative; border-radius:22px; overflow:hidden; min-height:430px; border:1px solid #4a0a0a;
        box-shadow:0 0 0 1px #000, 0 25px 70px rgba(225,6,0,.35); margin-bottom:22px; }
.hero-img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; }
.hero-overlay { position:absolute; inset:0; background:linear-gradient(90deg, rgba(0,0,0,.92) 0%, rgba(0,0,0,.65) 45%, rgba(138,0,0,.25) 100%); }
.hero-content { position:relative; z-index:2; padding:60px 55px; max-width:760px; }
.hero-tag { display:inline-block; background:var(--red); color:#fff; font-weight:800; font-size:13px; letter-spacing:1.5px; padding:8px 16px; border-radius:999px; box-shadow:0 0 25px rgba(225,6,0,.7); }
.hero-title { font-size:78px; font-weight:900; line-height:1; margin:22px 0 14px; color:#fff !important; letter-spacing:-2px; text-shadow:0 4px 30px rgba(0,0,0,.9); }
.hero-title span { color:var(--red); text-shadow:0 0 40px rgba(225,6,0,.8); }
.hero-sub { font-size:19px; color:#f1f1f1 !important; font-weight:500; line-height:1.6; text-shadow:0 2px 10px #000; }
.hero-chips { margin-top:24px; display:flex; flex-wrap:wrap; gap:10px; }
.hero-chips span { background:rgba(0,0,0,.65); border:1px solid var(--red); color:#fff; padding:9px 16px; border-radius:10px; font-weight:700; font-size:14px; }

/* ---------- MARQUEE (défilement des voitures) ---------- */
.section-title { color:#fff; font-size:22px; font-weight:900; padding:14px 20px; margin:28px 0 16px; background:linear-gradient(90deg, rgba(225,6,0,.35), transparent); border-left:6px solid var(--red); border-radius:10px; letter-spacing:.3px; }
.marquee { overflow:hidden; padding:6px 0 14px; -webkit-mask-image:linear-gradient(90deg, transparent, #000 6%, #000 94%, transparent); mask-image:linear-gradient(90deg, transparent, #000 6%, #000 94%, transparent); }
.mq-track { display:flex; gap:18px; width:max-content; animation: mqscroll 45s linear infinite; }
.marquee:hover .mq-track { animation-play-state:paused; }
@keyframes mqscroll { from { transform:translateX(0); } to { transform:translateX(-50%); } }
.mq-card { width:270px; flex:0 0 auto; background:linear-gradient(160deg,#1b1b1b,#0d0d0d); border:1px solid var(--line); border-bottom:3px solid var(--red); border-radius:16px; padding:14px; transition:transform .25s, box-shadow .25s; }
.mq-card:hover { transform:translateY(-6px); box-shadow:0 14px 34px rgba(225,6,0,.45); }
.mq-art { background:radial-gradient(circle at 50% 100%, rgba(225,6,0,.35), transparent 70%), #0a0a0a; border-radius:12px; padding:14px 10px 6px; }
.mq-art img { width:100%; display:block; }
.mq-photo { width:100%; height:150px; object-fit:cover; border-radius:12px; display:block; }
.hero-collage { position:absolute; inset:0; display:flex; }
.hero-collage img { flex:1; min-width:0; height:100%; object-fit:cover; filter:saturate(1.1) contrast(1.05); }
.hero-fallback { position:absolute; inset:0; background:radial-gradient(800px 400px at 75% 40%, #7a0000, #000 70%); }
.mq-ph { height:150px; display:flex; align-items:center; justify-content:center; font-size:48px; background:#0a0a0a; border-radius:12px; }
.mq-show { width:300px; }
.mq-reverse { animation-direction:reverse; animation-duration:70s; }
.mq-brand { width:230px; }
.mq-brand .mq-name { text-align:center; margin-bottom:2px; }
.photo-credit { color:var(--mut); font-size:12px; text-align:right; margin-top:-6px; }
.mq-name { color:#fff; font-weight:900; font-size:17px; margin-top:12px; }
.mq-meta { color:var(--mut); font-size:13px; margin-top:2px; }
.mq-price { color:var(--red2); font-weight:900; font-size:19px; margin-top:6px; }
.mq-foot { display:flex; justify-content:space-between; align-items:center; margin-top:10px; }
.mq-conf { color:#fff; font-weight:800; font-size:13px; }
.chip { font-size:12px; font-weight:800; padding:4px 11px; border-radius:999px; color:#fff; }
.s-top { background:#0f8a3c; } .s-good { background:#1769aa; } .s-mid { background:#c77700; } .s-low { background:#b30000; }

/* ---------- KPI ---------- */
.kpi-container { display:grid; grid-template-columns:repeat(4,1fr); gap:18px; margin-bottom:20px; }
.kpi-card { background:linear-gradient(160deg,#1c1c1c,#0b0b0b); border:1px solid var(--line); border-top:5px solid var(--red); border-radius:16px; padding:22px; min-height:150px; box-shadow:0 8px 28px rgba(0,0,0,.6), 0 0 22px rgba(225,6,0,.12); }
.kpi-icon { font-size:28px; }
.kpi-title { color:#d0d0d0; font-size:12px; font-weight:800; letter-spacing:1px; margin-top:6px; }
.kpi-value { color:#fff; font-size:38px; font-weight:900; margin-top:4px; text-shadow:0 0 18px rgba(225,6,0,.55); }
.kpi-sub { color:var(--red2); font-size:13px; font-weight:800; margin-top:4px; }

/* ---------- PANELS / BARRES ---------- */
.panel, .info-card, .legend-box, .result-card { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:22px; box-shadow:0 8px 24px rgba(0,0,0,.5); }
.panel-title, .info-card h3 { color:#fff !important; font-size:18px; font-weight:900; margin:0 0 16px; }
.info-card p { color:#d6d6d6 !important; font-size:14.5px; line-height:1.7; }
.bar-row { display:grid; grid-template-columns:120px 1fr 40px; gap:12px; align-items:center; margin-bottom:12px; }
.bar-label { color:#fff; font-weight:700; font-size:14px; }
.bar-track { background:#262626; border-radius:999px; height:14px; overflow:hidden; }
.bar-fill { height:100%; background:linear-gradient(90deg, var(--dred), var(--red2)); border-radius:999px; box-shadow:0 0 14px rgba(225,6,0,.7); }
.bar-val { color:#fff; font-weight:900; text-align:right; }
.legend-item { display:inline-block; color:#e0e0e0; font-size:14px; margin-right:24px; margin-bottom:6px; }
.legend-title { color:#fff; font-weight:900; margin-bottom:10px; }
.empty { color:var(--mut); padding:20px; text-align:center; }

/* ---------- TABLE ---------- */
.table-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:16px; background:var(--card); }
.dt { width:100%; border-collapse:collapse; font-size:14px; }
.dt thead th { background:linear-gradient(90deg, var(--dred), var(--red)); color:#fff; text-align:left; padding:14px 16px; font-weight:800; letter-spacing:.4px; white-space:nowrap; }
.dt td { padding:13px 16px; color:#f2f2f2; border-bottom:1px solid #242424; white-space:nowrap; }
.dt tbody tr:hover { background:rgba(225,6,0,.12); }
.t-strong { font-weight:800; color:#fff; }
.badge { padding:4px 12px; border-radius:999px; font-weight:800; font-size:13px; border:1px solid; }
.b-green { color:#4ade80; border-color:#4ade80; } .b-blue { color:#60a5fa; border-color:#60a5fa; }
.b-orange { color:#fbbf24; border-color:#fbbf24; } .b-red { color:#ff6b6b; border-color:#ff6b6b; }

/* ---------- RÉSULTAT ---------- */
.result-card { border-top:5px solid var(--red); text-align:center; }
.result-label { color:var(--mut); font-size:13px; font-weight:800; letter-spacing:1px; }
.result-value { color:#fff; font-size:34px; font-weight:900; margin:6px 0; text-shadow:0 0 20px rgba(225,6,0,.6); }
.result-conf { color:var(--red2); font-size:20px; font-weight:900; }

/* ---------- WIDGETS STREAMLIT ---------- */
.stTabs [data-baseweb="tab-list"] { gap:8px; background:#0d0d0d; padding:8px; border-radius:14px; border:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { background:transparent; border-radius:10px; color:#fff !important; font-weight:800; padding:10px 20px; }
.stTabs [aria-selected="true"] { background:linear-gradient(90deg, var(--dred), var(--red)) !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

.stButton > button, .stDownloadButton > button { background:#1a1a1a; color:#fff; border:1px solid var(--red); border-radius:12px; font-weight:800; padding:.6rem 1rem; transition:all .2s; }
.stButton > button:hover, .stDownloadButton > button:hover { background:var(--red); color:#fff; border-color:var(--red); box-shadow:0 0 22px rgba(225,6,0,.65); }
.stButton > button[kind="primary"] { background:linear-gradient(90deg, var(--dred), var(--red)); border:none; }

[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input { background:#151515 !important; color:#fff !important; border:1px solid #3a3a3a !important; border-radius:10px !important; }
[data-baseweb="select"] > div:hover, .stNumberInput input:focus { border-color:var(--red) !important; }
[data-baseweb="select"] svg { fill:#fff; }
[data-baseweb="popover"] li, [data-baseweb="menu"] { background:#151515 !important; color:#fff !important; }
[data-testid="stFileUploader"] section { background:#121212; border:2px dashed var(--red); border-radius:14px; }
[data-testid="stFileUploader"] * { color:#fff !important; }
[data-testid="stAlert"] { background:#1a1a1a; border:1px solid var(--red); border-radius:12px; }
[data-testid="stAlert"] * { color:#fff !important; }

.footer { text-align:center; color:var(--mut); padding:30px 0 10px; font-size:13px; }
.footer b { color:var(--red2); }

/* ---------- RESPONSIVE ---------- */
@media (max-width:1000px) { .kpi-container { grid-template-columns:repeat(2,1fr); } .hero-title { font-size:56px; } .hero-content { padding:40px 28px; } }
@media (max-width:600px) { .kpi-container { grid-template-columns:1fr; } .hero-title { font-size:40px; } .hero { min-height:380px; } .hero-sub { font-size:16px; } .bar-row { grid-template-columns:90px 1fr 34px; } }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ================================================================
# 7. HERO + DÉFILEMENT DES VOITURES
# ================================================================

st.markdown(build_hero(), unsafe_allow_html=True)

st.markdown('<div class="section-title">💎 Modèles d\'exception</div>', unsafe_allow_html=True)
st.markdown(build_showcase_marquee(), unsafe_allow_html=True)

st.markdown('<div class="section-title">🚘 Véhicules récemment analysés</div>', unsafe_allow_html=True)
st.markdown(build_marquee(st.session_state.prediction_history), unsafe_allow_html=True)

st.markdown('<div class="section-title">🏁 Toutes les marques</div>', unsafe_allow_html=True)
st.markdown(build_brand_marquee(), unsafe_allow_html=True)
st.markdown(
    '<div class="photo-credit">Photos : Wikipédia / Wikimedia Commons</div>',
    unsafe_allow_html=True,
)


# ================================================================
# 8. ONGLETS
# ================================================================

tab_dashboard, tab_single, tab_csv = st.tabs(
    ["📊 Tableau de bord", "🔍 Prédiction simple", "📁 Importation CSV"]
)

# ------------------------------------------------------------------
# ONGLET 1 : DASHBOARD
# ------------------------------------------------------------------

with tab_dashboard:
    st.markdown(build_kpis(st.session_state.prediction_history), unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Actualiser le tableau de bord", use_container_width=True):
            st.rerun()
    with col_b:
        if st.button("🗑️ Réinitialiser l'historique", use_container_width=True):
            st.session_state.prediction_history = pd.DataFrame(columns=HISTORY_COLUMNS)
            st.rerun()

    st.markdown('<div class="section-title">📈 Analyse visuelle</div>', unsafe_allow_html=True)
    chart1, chart2 = build_charts(st.session_state.prediction_history)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(chart1, unsafe_allow_html=True)
    with c2:
        st.markdown(chart2, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎯 Performance et qualité des prédictions</div>', unsafe_allow_html=True)
    i1, i2 = st.columns(2)
    with i1:
        st.markdown(_h("""
        <div class="info-card"><h3>🤖 Modèle de Machine Learning</h3>
        <p>Le système utilise un modèle <b>Random Forest</b> pour prédire automatiquement
        l'état probable d'un véhicule.</p>
        <p><b>Variables utilisées :</b><br>Marque • Transmission • Quartier • Année • Prix</p></div>
        """), unsafe_allow_html=True)
    with i2:
        st.markdown(_h("""
        <div class="info-card"><h3>🛡️ Niveau de fiabilité</h3>
        <p>Le score de confiance représente le niveau de certitude du modèle pour
        chaque prédiction.</p><p><b>Seuil recommandé :</b> 80 %</p></div>
        """), unsafe_allow_html=True)

    st.markdown(_h("""
    <div class="legend-box" style="margin-top:16px">
      <div class="legend-title">📌 Interprétation du score de confiance</div>
      <span class="legend-item">🟢 <b>≥ 90 %</b> — Très fiable</span>
      <span class="legend-item">🔵 <b>80–89 %</b> — Fiable</span>
      <span class="legend-item">🟠 <b>70–79 %</b> — À vérifier</span>
      <span class="legend-item">🔴 <b>&lt; 70 %</b> — Faible confiance</span>
    </div>
    """), unsafe_allow_html=True)

    st.markdown('<div class="section-title">🚗 Historique des dernières prédictions</div>', unsafe_allow_html=True)
    st.markdown(build_table(st.session_state.prediction_history), unsafe_allow_html=True)

    st.markdown('<div class="section-title">💡 Synthèse décisionnelle</div>', unsafe_allow_html=True)
    st.markdown(_h("""
    <div class="info-card"><h3>Interprétation du tableau de bord</h3>
    <p>L'application automatise l'évaluation de l'état des véhicules à partir de leurs
    caractéristiques. Les chiffres reflètent les prédictions réellement effectuées via les onglets
    « Prédiction simple » et « Importation CSV ».</p>
    <p>Un score élevé indique une forte certitude du modèle. Les résultats à faible score
    doivent être vérifiés avant toute décision.</p></div>
    """), unsafe_allow_html=True)

# ------------------------------------------------------------------
# ONGLET 2 : PRÉDICTION SIMPLE
# ------------------------------------------------------------------

with tab_single:
    st.markdown('<div class="section-title">🔍 Évaluation individuelle d\'un véhicule</div>', unsafe_allow_html=True)
    st.write("Renseignez les caractéristiques du véhicule pour obtenir une prédiction automatique.")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        marque = st.selectbox("🚗 Marque", MARQUES, key="marque_in")
        c1, c2 = st.columns(2)
        with c1:
            transmission = st.selectbox("⚙️ Transmission", TRANSMISSIONS, key="trans_in")
        with c2:
            quartier = st.selectbox("📍 Quartier", QUARTIERS, key="quartier_in")
        c3, c4 = st.columns(2)
        with c3:
            annee = st.number_input("📅 Année", value=2018, step=1, format="%d", key="annee_in")
        with c4:
            prix = st.number_input("💰 Prix (FCFA)", value=4500000, step=100000, key="prix_in")
        launch = st.button("🚀 Lancer la prédiction", type="primary", use_container_width=True)

    with col_right:
        if launch:
            try:
                prediction, confidence = predict_single(marque, transmission, quartier, annee, prix)
                _add_to_history(marque, annee, prix, quartier, prediction, confidence)
                st.markdown(_h(f"""
                <div class="result-card">
                  <div class="result-label">ÉTAT ÉVALUÉ</div>
                  <div class="result-value">{escape(prediction)}</div>
                  <div class="result-label">SCORE DE CONFIANCE</div>
                  <div class="result-conf">{confidence * 100:.1f}%</div>
                </div>
                """), unsafe_allow_html=True)
            except ValueError as e:
                st.error(f"❌ Erreur : {e}")
        else:
            st.markdown(_h("""
            <div class="result-card">
              <div class="result-label">RÉSULTAT</div>
              <div class="result-value">—</div>
              <div class="result-label">Lancez la prédiction pour voir l'état estimé.</div>
            </div>
            """), unsafe_allow_html=True)

# ------------------------------------------------------------------
# ONGLET 3 : IMPORTATION CSV
# ------------------------------------------------------------------

with tab_csv:
    st.markdown('<div class="section-title">📁 Analyse prédictive en masse</div>', unsafe_allow_html=True)
    st.write("Importez un fichier CSV contenant les colonnes :")
    st.markdown("**Marque • Transmission • Quartier • Annee • Prix**")

    uploaded_file = st.file_uploader("📤 Importer votre fichier CSV", type=["csv"])

    if st.button("🚀 Traiter le fichier", type="primary"):
        if uploaded_file is None:
            st.warning("⚠️ Veuillez importer un fichier CSV.")
        else:
            try:
                df_in = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"❌ Impossible de lire le CSV : {e}")
                df_in = None

            if df_in is not None:
                df_out, message = predict_csv(df_in)
                if df_out is None:
                    st.error(message)
                else:
                    st.success(message)
                    st.dataframe(df_out, use_container_width=True, hide_index=True)
                    st.download_button(
                        "📥 Télécharger les résultats",
                        data=df_out.to_csv(index=False).encode("utf-8"),
                        file_name="predictions_resultats.csv",
                        mime="text/csv",
                    )


# ================================================================
# 9. FOOTER
# ================================================================

st.markdown(
    '<div class="footer">© CoinAfrique AI • Prédiction intelligente de l\'état des véhicules • '
    '<b>Rouge &amp; Noir Edition</b></div>',
    unsafe_allow_html=True,
)