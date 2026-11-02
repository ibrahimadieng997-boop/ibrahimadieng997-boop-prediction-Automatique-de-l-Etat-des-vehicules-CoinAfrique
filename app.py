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

import datetime as dt
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="CoinAfrique AI | Vehicle Intelligence",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 2. IDENTITE VISUELLE
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #F4F6FA;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: #101D35;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

.main-title {
    background: linear-gradient(120deg,#101D35,#193B67);
    padding: 32px;
    border-radius: 20px;
    border: 1px solid #D7B56D;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(16,29,53,.15);
}

.main-title h1 {
    color: #FFFFFF;
    font-size: 32px;
    font-weight: 800;
    margin: 0;
}

.main-title p {
    color: #D9E2F2;
    font-size: 15px;
    margin-top: 10px;
}

.gold { color: #E4BE73; }

.section-title {
    color: #142744;
    font-size: 22px;
    font-weight: 800;
    padding: 12px 0;
    border-bottom: 2px solid #D7B56D;
    margin: 20px 0;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #E2E7EF;
    box-shadow: 0 4px 15px rgba(16,29,53,.05);
}

div[data-testid="stMetricLabel"] {
    color: #64748B;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #142744;
    font-weight: 800;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(120deg,#142744,#24548A);
    color: white;
    border: 1px solid #D7B56D;
    border-radius: 10px;
    min-height: 48px;
    font-weight: 700;
    transition: .2s;
}

div.stButton > button[kind="primary"]:hover {
    background: #D7B56D;
    color: #142744;
}

.result-card {
    background: #FFFFFF;
    padding: 25px;
    border-radius: 16px;
    border-left: 6px solid #D7B56D;
    box-shadow: 0 5px 20px rgba(16,29,53,.07);
    margin: 15px 0;
}

.result-label {
    color: #64748B;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
}

.result-value {
    color: #142744;
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}

.info-box {
    background: #EAF0F8;
    border-radius: 12px;
    padding: 17px;
    color: #142744;
    border: 1px solid #D6E0EE;
}

.footer {
    background: #101D35;
    color: #D9E2F2;
    text-align: center;
    padding: 20px;
    border-radius: 12px;
    margin-top: 35px;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 700;
    color: #142744;
}

.stTabs [aria-selected="true"] {
    color: #A67C32 !important;
}

@media(max-width:700px) {
    .main-title h1 {font-size:23px;}
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. CHARGEMENT DU MODELE
# ==========================================================

@st.cache_resource
def load_artifacts():

    encoders = joblib.load("encoders.joblib")
    var_cat_vid = joblib.load("var_cat_vid.joblib")
    scaler = joblib.load("scaler.joblib")
    rf = joblib.load("rf_model.joblib")

    return encoders, var_cat_vid, scaler, rf


try:
    encoders, var_cat_vid, scaler, rf = load_artifacts()

except Exception as e:
    st.error(
        "Erreur de chargement des fichiers du modèle. "
        "Vérifie leur présence dans le dépôt Streamlit."
    )
    st.exception(e)
    st.stop()


MARQUES = list(var_cat_vid[0])
TRANSMISSIONS = list(var_cat_vid[1])
QUARTIERS = list(var_cat_vid[2])
CLASS_NAMES = list(var_cat_vid[3])


# ==========================================================
# 4. INITIALISATION DE L'HISTORIQUE
# ==========================================================

HISTORY_COLUMNS = [
    "Véhicule",
    "Année",
    "Prix",
    "Quartier",
    "État prédit",
    "Confiance",
    "Date"
]

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=HISTORY_COLUMNS
    )


def format_price(value):

    return f"{float(value):,.0f} FCFA".replace(",", " ")


def add_history(marque, annee, prix, quartier, etat, confidence):

    row = {
        "Véhicule": str(marque),
        "Année": int(annee),
        "Prix": format_price(prix),
        "Quartier": str(quartier),
        "État prédit": str(etat),
        "Confiance": round(confidence * 100, 2),
        "Date": dt.datetime.now()
    }

    new_row = pd.DataFrame([row])

    st.session_state.history = pd.concat(
        [st.session_state.history, new_row],
        ignore_index=True
    )


# ==========================================================
# 5. CORRESPONDANCE DES CLASSES
# ==========================================================

def class_name(label):

    # Correspondance avec les classes enregistrées
    if isinstance(label, (int, np.integer)):
        index = int(label)

        if 0 <= index < len(CLASS_NAMES):
            return str(CLASS_NAMES[index])

    return str(label)


# ==========================================================
# 6. MOTEUR DE PREDICTION
# ==========================================================

def predict_single(marque, transmission, quartier, annee, prix):

    annee = float(annee)
    prix = float(prix)

    if not np.isfinite(annee) or not np.isfinite(prix):
        raise ValueError("Les valeurs numériques sont invalides.")

    if annee < 1900 or annee > dt.date.today().year + 1:
        raise ValueError("Année de fabrication invalide.")

    if prix <= 0:
        raise ValueError("Le prix doit être supérieur à zéro.")

    try:
        marque_enc = encoders[0].transform([marque])[0]
        trans_enc = encoders[1].transform([transmission])[0]
        quartier_enc = encoders[2].transform([quartier])[0]

    except ValueError as e:
        raise ValueError(
            f"Catégorie non reconnue par le modèle : {e}"
        )

    # Respecter exactement l'ordre des variables d'entraînement
    X = np.array([[
        marque_enc,
        annee,
        trans_enc,
        prix,
        quartier_enc
    ]], dtype=float)

    X_scaled = scaler.transform(X)

    prediction = rf.predict(X_scaled)[0]
    probabilities = rf.predict_proba(X_scaled)[0]

    etat = class_name(prediction)

    classes = [
        class_name(label) for label in rf.classes_
    ]

    probability_dict = {
        classes[i]: float(probabilities[i])
        for i in range(len(classes))
    }

    confidence = float(np.max(probabilities))

    return etat, confidence, probability_dict


# ==========================================================
# 7. EN-TETE
# ==========================================================

st.markdown("""
<div class="main-title">
    <h1>🚘 COINA<span class="gold">FRIQUE</span> AI</h1>
    <p>
        VEHICLE INTELLIGENCE PLATFORM |
        Machine Learning • Random Forest • Predictive Analytics
    </p>
    <p>
        Plateforme intelligente d'évaluation automatique
        de l'état des véhicules.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# 8. MENU LATERAL
# ==========================================================

with st.sidebar:

    st.markdown("# 🚘 COINAFFRIQUE")
    st.markdown("### AI Intelligence Center")

    st.divider()

    st.markdown("**Architecture du système**")

    st.success("Modèle Random Forest chargé")

    st.write("**Variables utilisées**")
    st.caption("Marque")
    st.caption("Année")
    st.caption("Transmission")
    st.caption("Prix")
    st.caption("Quartier")

    st.divider()

    st.markdown("**Informations**")
    st.write("Version : 2.0")
    st.write("Moteur : Random Forest")
    st.write("Application : Streamlit")

    st.divider()

    st.caption("© 2026 CoinAfrique AI")


# ==========================================================
# 9. ONGLETS
# ==========================================================

tab_dashboard, tab_prediction, tab_batch, tab_about = st.tabs([
    "📊 Dashboard",
    "🔍 Prédiction intelligente",
    "📁 Analyse CSV",
    "ℹ️ À propos"
])


# ==========================================================
# 10. DASHBOARD
# ==========================================================

with tab_dashboard:

    st.markdown(
        '<div class="section-title">Vue globale des performances</div>',
        unsafe_allow_html=True
    )

    history = st.session_state.history.copy()

    total = len(history)

    if total > 0:

        avg_conf = history["Confiance"].mean()

        avg_price = (
            history["Prix"]
            .str.replace(" FCFA", "", regex=False)
            .str.replace(" ", "", regex=False)
            .astype(float)
            .mean()
        )

        today = dt.date.today()

        today_count = (
            pd.to_datetime(history["Date"]).dt.date == today
        ).sum()

    else:

        avg_conf = 0
        avg_price = 0
        today_count = 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Prédictions réalisées",
        total
    )

    c2.metric(
        "Confiance moyenne",
        f"{avg_conf:.1f}%"
    )

    c3.metric(
        "Prix moyen analysé",
        f"{avg_price / 1_000_000:.2f} M FCFA"
    )

    c4.metric(
        "Prédictions aujourd'hui",
        int(today_count)
    )

    st.markdown(
        '<div class="section-title">Analyse des résultats</div>',
        unsafe_allow_html=True
    )

    if total > 0:

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### Répartition des états prédits")

            distribution = (
                history["État prédit"]
                .value_counts()
            )

            st.bar_chart(distribution)

        with col2:

            st.markdown("#### Évolution des prédictions")

            evolution = history.copy()

            evolution["Date"] = pd.to_datetime(
                evolution["Date"]
            )

            evolution = (
                evolution.set_index("Date")
                .resample("D")
                .size()
            )

            st.line_chart(evolution)

        st.markdown(
            '<div class="section-title">Historique des analyses</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            history.sort_values(
                "Date", ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        csv_history = history.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "📥 Télécharger l'historique",
            csv_history,
            "historique_coinAfrique.csv",
            "text/csv"
        )

    else:

        st.info(
            "Aucune prédiction réelle pour le moment. "
            "Commence par analyser un véhicule."
        )

    if st.button("🗑️ Effacer l'historique"):

        st.session_state.history = pd.DataFrame(
            columns=HISTORY_COLUMNS
        )

        st.rerun()


# ==========================================================
# 11. PREDICTION INDIVIDUELLE
# ==========================================================

with tab_prediction:

    st.markdown(
        '<div class="section-title">Évaluation intelligente d’un véhicule</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "Complète les informations du véhicule pour lancer "
        "une analyse avec notre modèle Random Forest."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            marque = st.selectbox(
                "🚗 Marque du véhicule",
                MARQUES
            )

            transmission = st.selectbox(
                "⚙️ Type de transmission",
                TRANSMISSIONS
            )

            quartier = st.selectbox(
                "📍 Quartier",
                QUARTIERS
            )

        with col2:

            annee = st.number_input(
                "📅 Année de fabrication",
                min_value=1900,
                max_value=dt.date.today().year + 1,
                value=2018,
                step=1
            )

            prix = st.number_input(
                "💰 Prix du véhicule (FCFA)",
                min_value=1,
                value=4500000,
                step=100000
            )

        submitted = st.form_submit_button(
            "🚀 LANCER L'ANALYSE IA",
            type="primary",
            use_container_width=True
        )

    if submitted:

        try:

            etat, confidence, probabilities = predict_single(
                marque,
                transmission,
                quartier,
                annee,
                prix
            )

            add_history(
                marque,
                annee,
                prix,
                quartier,
                etat,
                confidence
            )

            st.session_state.last_result = {
                "etat": etat,
                "confidence": confidence,
                "probabilities": probabilities,
                "marque": marque,
                "annee": annee,
                "prix": prix
            }

        except Exception as e:

            st.error(f"Erreur pendant la prédiction : {e}")

    if "last_result" in st.session_state:

        result = st.session_state.last_result

        st.markdown(
            "<div class='section-title'>Résultat de l'analyse IA</div>",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">État prédit</div>
                    <div class="result-value">
                        {result["etat"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Score de confiance estimé</div>
                    <div class="result-value">
                        {result["confidence"] * 100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("#### Probabilités par classe")

        prob_df = pd.DataFrame({
            "État": list(result["probabilities"].keys()),
            "Probabilité (%)": [
                v * 100
                for v in result["probabilities"].values()
            ]
        }).sort_values(
            "Probabilité (%)",
            ascending=False
        )

        st.bar_chart(
            prob_df.set_index("État")
        )

        for _, row in prob_df.iterrows():

            st.write(
                f"**{row['État']} : {row['Probabilité (%)']:.2f}%**"
            )

            st.progress(
                float(row["Probabilité (%)"] / 100)
            )

        st.markdown(
            """
            <div class="info-box">
                <b>Interprétation :</b>
                les probabilités indiquent les scores attribués
                par le modèle aux différentes classes.
                Elles ne constituent pas une garantie de l'état
                mécanique réel du véhicule.
                Une inspection physique reste indispensable.
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================================
# 12. PREDICTION EN MASSE CSV
# ==========================================================

def find_column(df, name):

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    return normalized.get(name.lower())


def predict_csv(df):

    required = [
        "Marque",
        "Transmission",
        "Quartier",
        "Annee",
        "Prix"
    ]

    # Normalisation des noms de colonnes
    rename_map = {}

    aliases = {
        "marque": "Marque",
        "transmission": "Transmission",
        "quartier": "Quartier",
        "annee": "Annee",
        "année": "Annee",
        "prix": "Prix"
    }

    for col in df.columns:

        key = str(col).strip().lower()

        if key in aliases:
            rename_map[col] = aliases[key]

    df = df.rename(columns=rename_map)

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Colonnes manquantes : " + ", ".join(missing)
        )

    if len(df) == 0:
        raise ValueError("Le fichier CSV est vide.")

    if len(df) > 5000:
        raise ValueError(
            "Maximum autorisé : 5 000 lignes par importation."
        )

    results = []
    history_rows = []

    for index, row in df.iterrows():

        try:

            etat, confidence, probabilities = predict_single(
                row["Marque"],
                row["Transmission"],
                row["Quartier"],
                row["Annee"],
                row["Prix"]
            )

            results.append({
                "Etat_Predit": etat,
                "Score_Confiance": round(
                    confidence * 100, 2
                ),
                "Statut": "Succès"
            })

            history_rows.append({
                "Véhicule": str(row["Marque"]),
                "Année": int(float(row["Annee"])),
                "Prix": format_price(row["Prix"]),
                "Quartier": str(row["Quartier"]),
                "État prédit": etat,
                "Confiance": round(confidence * 100, 2),
                "Date": dt.datetime.now()
            })

        except Exception as e:

            results.append({
                "Etat_Predit": "Erreur",
                "Score_Confiance": 0,
                "Statut": str(e)
            })

    result_df = pd.concat(
        [
            df.reset_index(drop=True),
            pd.DataFrame(results)
        ],
        axis=1
    )

    if history_rows:

        st.session_state.history = pd.concat(
            [
                st.session_state.history,
                pd.DataFrame(history_rows)
            ],
            ignore_index=True
        )

    return result_df


with tab_batch:

    st.markdown(
        '<div class="section-title">Analyse prédictive en masse</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Importe un fichier CSV contenant les caractéristiques "
        "des véhicules."
    )

    st.markdown("""
    **Colonnes obligatoires :**

    `Marque | Transmission | Quartier | Annee | Prix`
    """)

    sample = pd.DataFrame({
        "Marque": [MARQUES[0]],
        "Transmission": [TRANSMISSIONS[0]],
        "Quartier": [QUARTIERS[0]],
        "Annee": [2018],
        "Prix": [4500000]
    })

    st.download_button(
        "📥 Télécharger un modèle CSV",
        sample.to_csv(index=False).encode("utf-8-sig"),
        "modele_coinAfrique.csv",
        "text/csv"
    )

    uploaded = st.file_uploader(
        "Importer un fichier CSV",
        type=["csv"]
    )

    if uploaded is not None:

        if st.button(
            "🚀 ANALYSER LES VÉHICULES",
            type="primary",
            use_container_width=True
        ):

            try:

                uploaded.seek(0)

                df = pd.read_csv(
                    uploaded,
                    sep=None,
                    engine="python"
                )

                with st.spinner(
                    "Analyse des véhicules en cours..."
                ):

                    output = predict_csv(df)

                success_count = (
                    output["Statut"] == "Succès"
                ).sum()

                st.success(
                    f"{success_count} prédictions réussies "
                    f"sur {len(output)} véhicules."
                )

                st.dataframe(
                    output,
                    use_container_width=True,
                    hide_index=True
                )

                csv_result = output.to_csv(
                    index=False
                ).encode("utf-8-sig")

                st.download_button(
                    "📥 Télécharger les résultats",
                    csv_result,
                    "resultats_coinAfrique.csv",
                    "text/csv"
                )

            except Exception as e:

                st.error(f"Erreur : {e}")


# ==========================================================
# 13. A PROPOS
# ==========================================================

with tab_about:

    st.markdown(
        '<div class="section-title">Présentation du projet</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### 🚘 CoinAfrique AI

    Une plateforme de prédiction automatique de l'état des véhicules
    basée sur les techniques de Machine Learning.

    **Technologies utilisées**

    - Python
    - Pandas et NumPy
    - Scikit-learn
    - Random Forest
    - Joblib
    - Streamlit

    **Variables explicatives**

    - Marque du véhicule
    - Année de fabrication
    - Type de transmission
    - Prix
    - Quartier

    **Fonctionnalités**

    - Prédiction individuelle
    - Classification automatique
    - Estimation des probabilités par classe
    - Traitement de fichiers CSV
    - Tableau de bord analytique

    **Important :** ce système fournit une estimation statistique
    et ne remplace pas une expertise mécanique.
    """)

    st.info(
        "Ce projet constitue une application expérimentale "
        "de Machine Learning pour l'analyse du marché automobile."
    )


# ==========================================================
# 14. PIED DE PAGE
# ==========================================================

st.markdown("""
<div class="footer">
    <b>COINAFFRIQUE AI</b><br>
    Intelligent Vehicle Prediction Platform<br>
    Machine Learning • Data Science • Artificial Intelligence<br>
    © 2026
</div>
""", unsafe_allow_html=True)


