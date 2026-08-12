import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from api.application_model import ApplicationModel
from models.predictor import ApplicationRiskPredictor

load_dotenv()

api_key_header = APIKeyHeader(name="x-api-key")
predictor = ApplicationRiskPredictor()

def api_predict(application: ApplicationModel):
    result = predictor.predict(application)

    return {
        "sk_id_curr": application.SK_ID_CURR,
        "prediction": result['prediction'],
        "probability": result['probability'],
    }

app = FastAPI(
    title="Api de prédiction du risque d'attribution d'un crédit bancaire.",
    description="Cette API permet de prédire le risque d'attribution d'un crédit bancaire pour un demande donné.",
    version="0.0.1",
    openapi_tags=[
        {
            "name": "Prédiction du risque d'attribution d'un crédit bancaire",
            "description": "Endpoints pour prédire le risque d'attribution d'un crédit bancaire"
        }
    ],
    contact={
        "email": "aurelien.clugery@gmail.com",
    }
)

async def verify_api_key(api_key: str = Security(api_key_header)):

    key = os.getenv('API_KEY')

    if not key or api_key != key:
        raise HTTPException(status_code=403)

    return api_key

@app.get(
    "/health",
    summary="Vérifier la santé de l'API",
    responses={
        200: {"description": "Le serveur de l'API fonctionne correctement"},
    }
)
def health_endpoint():
    return {"message": "Alive !"}

@app.post(
    "/predict",
    dependencies=[Depends(verify_api_key)],
    summary="",
    description="",
    responses={
        200: {"description": "Prédiction réussie"},
        401: {"description": "Clé API manquante ou invalide"},
        422: {"description": "Données d'employé invalides (erreur de validation Pydantic)"}
    }
)
def predict_endpoint(application: ApplicationModel):

    prediction = api_predict(application)

    return prediction



