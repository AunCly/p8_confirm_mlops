from pathlib import Path

import joblib
import os
import pandas as pd
from dotenv import load_dotenv
import numpy as np

from api.application_model import ApplicationModel

def format_data_for_ml_model(application: ApplicationModel) -> pd.DataFrame:
    data = pd.DataFrame([application.model_dump()])

    # 2. Remplacer les None par np.nan pour que Pandas garde un type numérique (float)
    cleaned_data = {
        k: (v if v is not None else np.nan) for k, v in data.items()
    }

    # 3. Créer le DataFrame Pandas
    df = pd.DataFrame([cleaned_data])

    # 4. Convertir les booléens en entiers (0/1) ou s'assurer que tout est numérique
    # LightGBM gère parfaitement les floats avec des NaN pour les valeurs manquantes
    for col in df.columns:
        # Convertit les colonnes booléennes ou object en float/numeric
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def _load(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artefact introuvable : {path}")
    return joblib.load(path)

class ApplicationRiskPredictor:
    def __init__(self, model_name: str = "model"):

        load_dotenv()
        base_dir = Path(__file__).resolve().parent.parent

        self.model_path = base_dir / f"models/compiled/{model_name}.pkl"

        self.model = _load(self.model_path)

    def predict(self, application_data: "ApplicationModel") -> dict:
        df_input = format_data_for_ml_model(application_data)

        prediction = self.model.predict(df_input)
        probability = self.model.predict_proba(df_input)

        return {
            "prediction": int(prediction[0]),
            "probability": float(probability[0][1]),
        }