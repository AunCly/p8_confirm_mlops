import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from database import database_manager as database

st.set_page_config(
    page_title="Dashboard Suivi", page_icon="📈", layout="wide"
)

st.title("📊 Dashboard de Suivi")

predictions = database.get_predictions()
prod_application = [prediction['input_data'] for prediction in predictions]

prod_application = pd.DataFrame(prod_application)
prod_application = prod_application.apply(pd.to_numeric, errors="coerce").astype('float64')
prod_application.fillna(0, inplace=True)

# Création de 4 colonnes sur une seule ligne
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Nombre de prédictions : ", value=len(prod_application))
with col2:
    st.metric(label="Montant moyen demandés", value=prod_application['app_AMT_CREDIT'].mean())
with col3:
    st.metric(label="Temps d'inférence", value=f"{np.round(np.mean([prediction['inference_time'] for prediction in predictions]), 2)}s")

# Fonction de chargement du fichier JSON
@st.cache_data
def load_drift_report(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Chargement des données
try:
    base_path = Path(__file__).resolve().parent.parent
    report_data = load_drift_report(base_path / "notebooks" / "drift_report.json")
    metrics = report_data.get("metrics", [])

    # Extraction des variables et métriques
    global_count = 0
    global_share = 0.0
    column_drifts = []

    for m in metrics:
        metric_name = m.get("metric_name", "")

        # Récupération du compteur global de dérive
        if "DriftedColumnsCount" in metric_name:
            global_count = m["value"].get("count", 0)
            global_share = m["value"].get("share", 0.0)

        # Récupération des dérives individuelles par colonne (ValueDrift)
        elif "ValueDrift" in metric_name:
            col_name = m["config"].get("column")
            p_value = m["value"]
            threshold = m["config"].get("threshold", 0.05)
            is_drifted = p_value < threshold

            column_drifts.append({
                "Variable": col_name,
                "P-Value": p_value,
                "Seuil": threshold,
                "Drift Détecté": "Oui ⚠️" if is_drifted else "Non ✅",
            })

    # Conversion en DataFrame Pandas
    df_drift = pd.DataFrame(column_drifts)

    # --- Section 1 : KPIs Globaux ---
    st.subheader("Analyse du drift")
    col1, col2, col3 = st.columns(3)

    total_columns = len(df_drift)

    with col1:
        st.metric(label="Variables Analysées", value=total_columns)
    with col2:
        st.metric(
            label="Variables en Dérive",
            value=f"{int(global_count)} ({global_share:.0%})",
        )
    with col3:
        status_alert = (
            "Alerte Critique ⚠️" if global_share > 0.4 else "Modéré / Stable ✅"
        )
        st.metric(label="Statut du Dataset", value=status_alert)

    st.markdown("---")

    # --- Section 2 : Tableau Interactif ---
    st.subheader("Détail du Drift par Variable")

    # Filtre interactif pour l'utilisateur
    filtre = st.radio(
        "Filtrer les variables :",
        ["Toutes", "Uniquement en dérive", "Uniquement stables"],
        horizontal=True,
    )

    df_display = df_drift.copy()
    if filtre == "Uniquement en dérive":
        df_display = df_display[df_display["Drift Détecté"].str.contains("Oui")]
    elif filtre == "Uniquement stables":
        df_display = df_display[df_display["Drift Détecté"].str.contains("Non")]

    # Affichage du tableau propre
    st.dataframe(
        df_display.style.format(
            {"P-Value": "{:.2e}"}
        ),  # Format scientifique pour les p-values très basses
        use_container_width=True,
        hide_index=True,
    )

except FileNotFoundError:
    st.error(
        "Le fichier `drift_report.json` est introuvable. Assurez-vous qu'il est"
        " placé dans le même dossier que ce script."
    )
except Exception as e:
    st.error(f"Une erreur est survenue lors de l'analyse du fichier JSON : {e}")

