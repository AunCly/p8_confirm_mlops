from pathlib import Path
import os
import pandas as pd
import numpy as np
import onnxruntime as ort  # <-- Nouvelle dépendance à installer (pip install onnxruntime)
from dotenv import load_dotenv

from api.application_model import ApplicationModel


def format_data_for_ml_model(application: ApplicationModel) -> np.ndarray:
    # 1. On récupère le dictionnaire brut de Pydantic
    data_dict = application.model_dump()

    # 2. On crée le DataFrame (1 ligne, X colonnes)
    df = pd.DataFrame([data_dict])

    # 3. On convertit tout en numérique
    df = df.apply(pd.to_numeric, errors="coerce")

    # ONNX exige qu'on remplace les NaN par une valeur si votre modèle ne les gère pas,
    # mais LightGBM les gère souvent très bien.
    # df = df.fillna(0.0) # À décommenter si besoin

    # 4. MODIFICATION CRUCIALE : ONNX a besoin d'un tableau NumPy en float32
    return df.to_numpy(dtype=np.float32)


class OnnxApplicationRiskPredictor:
    def __init__(self, model_name: str = "model"):
        load_dotenv()
        base_dir = Path(__file__).resolve().parent.parent

        # On pointe désormais vers le fichier .onnx
        self.model_path = base_dir / f"models/compiled/{model_name}.onnx"

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Artefact introuvable : {self.model_path}")

        # Initialisation de la session d'inférence ONNX
        self.session = ort.InferenceSession(str(self.model_path))

        # On récupère dynamiquement le nom de la variable d'entrée attendue par le graphe ONNX
        # (souvent 'float_input' ou 'X', défini lors de la conversion)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, application_data: "ApplicationModel") -> dict:
        # Récupération du tableau numpy en float32 (dimension: [1, n_features])
        input_data = format_data_for_ml_model(application_data)

        # Inférence avec ONNX Runtime
        # session.run prend en paramètre (noms_des_sorties_voulues, {nom_entree: donnees})
        # None signifie "Renvoie toutes les sorties définies"
        onnx_outputs = self.session.run(None, {self.input_name: input_data})

        # Par défaut avec scikit-learn/LightGBM convertis, ONNX renvoie une liste de 2 éléments :
        # onnx_outputs[0] = Les labels prédits (array numpy)
        # onnx_outputs[1] = Les probabilités (une liste de dictionnaires)

        prediction_label = int(onnx_outputs[0][0])
        probabilities = onnx_outputs[1][0]  # On prend le dict de probas de la première (et unique) ligne

        # On récupère la probabilité de la classe 1 (le risque/positif)
        # Selon la configuration de conversion (zipmap), probabilities peut être un dict ou un array
        if isinstance(probabilities, dict):
            probability = float(probabilities.get(1, 0.0))  # Clé 1 pour la classe 1
        else:
            probability = float(probabilities[1])  # Index 1 pour la classe 1

        return {
            "prediction": prediction_label,
            "probability": probability,
        }