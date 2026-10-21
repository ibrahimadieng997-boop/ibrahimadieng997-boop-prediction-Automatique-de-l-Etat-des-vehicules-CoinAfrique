<div align="center">

# 🚗 CoinAfrique AI

### Prédire l'état d'un véhicule d'occasion grâce au Machine Learning

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)

</div>

---

## 🎯 Le projet en 30 secondes

Sur les plateformes d'annonces comme **CoinAfrique**, il est difficile de savoir si un véhicule est réellement en bon état.

**CoinAfrique AI** résout ce problème : à partir de 5 informations (marque, transmission, quartier, année, prix), un modèle **Random Forest** prédit l'**état du véhicule** et indique un **score de confiance** pour chaque résultat.

L'application est un **tableau de bord web interactif**, utilisable sans aucune connaissance technique.

## ✨ Ce que fait l'application

- 📊 **Tableau de bord** : indicateurs clés et historique des prédictions
- 🔍 **Prédiction simple** : on remplit un formulaire, on obtient l'état du véhicule en un clic
- 📁 **Prédiction en masse** : import d'un fichier CSV, résultats téléchargeables
- 🛡️ **Score de confiance** : 🟢 ≥ 90 % très fiable · 🔵 80-89 % fiable · 🟠 70-79 % à vérifier · 🔴 < 70 % faible

## 🧠 Démarche Data Science

| Étape | Ce qui a été réalisé |
|---|---|
| **Données** | Annonces de véhicules (marque, transmission, quartier, année, prix) |
| **Pré-traitement** | Encodage des variables catégorielles, normalisation des données |
| **Modélisation** | Classification avec un Random Forest |
| **Déploiement** | Application Streamlit avec interface personnalisée |

## 🛠️ Compétences démontrées

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `Machine Learning (classification)` · `Streamlit` · `Sérialisation de modèles (Joblib)` · `Interface utilisateur (HTML/CSS)` · `Traitement de fichiers CSV`

## 🚀 Lancer le projet
```
Lancer l'application
```
[Lien vers l'application: https://prediction-automatique-de-l-etat-des-vehicules-coinafrique.streamlit.app/]


## 📁 Format du CSV (import en masse)

```csv
Marque,Transmission,Quartier,Annee,Prix
Toyota Corolla,Automatique,Almadies,2019,5500000
```

Le fichier de sortie ajoute les colonnes `Etat_Predit` et `Score_Confiance`.

## 🗂️ Structure

```
├── app.py               # Application Streamlit
├── rf_model.joblib      # Modèle Random Forest entraîné
├── encoders.joblib      # Encodeurs catégoriels
├── scaler.joblib        # Normalisation
├── var_cat_vid.joblib   # Modalités et noms des classes
└── requirements.txt
```

## 👤 Auteur

**Ibrahima Dieng**
Junior en Data Science et Intelligence Artificielle

📍 Thiès, Médina Fall, Sénégal
💻 GitHub : [ibrahimadieng997boop](https://github.com/ibrahimadieng997boop)
✉️ [ibrahimadieng997@gmail.com](mailto:ibrahimadieng997@gmail.com)

---

<div align="center">

⭐ *Ce projet vous plaît ? Laissez une étoile, et n'hésitez pas à me contacter !*

</div>
