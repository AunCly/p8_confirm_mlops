from pathlib import Path

import joblib
import os
import pandas as pd
from dotenv import load_dotenv
import numpy as np

from api.application_model import ApplicationModel


def format_data_for_ml_model(application: ApplicationModel) -> pd.DataFrame:
    # 1. On récupère le dictionnaire brut de Pydantic
    data_dict = application.model_dump()

    # 2. On crée le DataFrame (1 ligne, X colonnes)
    df = pd.DataFrame([data_dict])

    # 3. On convertit tout en numérique (les booléens deviennent 0/1,
    # et les None ou valeurs invalides deviennent automatiquement des np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")

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