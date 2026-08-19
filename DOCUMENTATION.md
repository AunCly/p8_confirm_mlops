# Documentation Technique — API REST de classification du risque lors de l'attribution d'un crédit bancaire

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Stack technique](#2-stack-technique)
3. [Architecture du projet](#3-architecture-du-projet)
4. [Installation et démarrage](#4-installation-et-démarrage)
5. [Variables d'environnement](#5-variables-denvironnement)
6. [Base de données](#6-base-de-données)
7. [API — Endpoints](#7-api--endpoints)
8. [Modèle de Machine Learning](#8-modèle-de-machine-learning)
9. [Tests](#9-tests)
10. [CI/CD et déploiement](#10-cicd-et-déploiement)
11. [Notebooks d'exploration](#11-notebooks-dexploration)

---

## 1. Vue d'ensemble

Ce projet est une **API REST de classification du risque lors de l'attribution d'un crédit bancaire**, développée dans le cadre du projet "Initiez-vous au MLOps 1/2".

Un modèle LGBM entraîné sur des données de la demande, mais aussi les données bancaires des précédents crédit demandé y compris dans d'autres organismes bancaire. Pour demande soumise, l'API retourne :
- une **prédiction binaire** (0 = reste, 1 = risque de départ)
- une **probabilité** associée (0.0 → 1.0)

L'historique de toutes les prédictions est persisté en dans un fichier de log predictions.log 

---

## 2. Stack technique

| Catégorie | Technologie             | Version |
|---|-------------------------|---|
| Langage | Python                  | 3.13 |
| Framework API | FastAPI                 | ≥ 0.138.1 |
| Machine Learning | scikit-learn            | ≥ 1.9.0 |
| Sérialisation modèle | onnx                    | ≥ 1.5.3 |
| Rééchantillonnage | imbalanced-learn        | ≥ 0.0 |
| Manipulation données | pandas                  | ≥ 3.0.3 |
| Validation données | Pydantic (via FastAPI)  | — |
| Client HTTP (tests) | httpx2                  | ≥ 2.5.0 |
| Tests | pytest                  | ≥ 9.1.1 |
| Conteneurisation | Docker + Docker Compose | — |
| Gestionnaire de paquets | uv                      | — |
| Stockage modèles (Git) | Git LFS                 | — |
| CI/CD | GitHub Actions          | — |
| Notebooks | Jupyter                 | ≥ 1.1.1 |

---

## 3. Architecture du projet

- .github/workflows/ : pipelines CI/CD GitHub Actions
- data/raw/ : fichiers CSV bruts (SIRH, évaluations, sondages
- database/ : fichier de log predictions.log
- models/ : code du modèle de ML, pipeline de préprocessing et utilitaires
- notebooks/ : notebooks Jupyter d'exploration et d'analyse des données
- api/ : code de l'API FastAPI (endpoints, validation, préprocessing, prédiction)
- dashboard/ : code du dashboard de monitoring (Streamlit)
- tests/ : tests unitaires et fonctionnels avec Pytest
- .env.example : template des variables d'environnement
- docker-compose.yml : configuration des services Docker (API + BDD)
- Dockerfile : image Docker de l'API FastAPI
- employee.py : modèle Pydantic pour la validation des données d'entrée
- pyproject.toml : déclaration des dépendances et configuration du projet
- README.md : présentation du projet et instructions d'installation
- DOCUMENTATION.md : documentation technique détaillée (ce fichier)

### Flux de données

1. Le client envoie une requête POST à l'endpoint `/predict` avec les données d'un employé. 
2. Les données sont validées par Pydantic (modèle `ApplicationModel`). 
4. Le modèle LGBM pré-entraîné effectue la prédiction et calcule la probabilité 
5. La prédiction est sauvegardée dans un fichier de log `predictions.log` pour l'historique.
6. L'API retourne une réponse JSON contenant l'`sk_id_curr`, la `prediction` et la `probability`.

## 4. Installation et démarrage

### Prérequis

- Docker et Docker Compose installés
- `uv` installé (`pip install uv`) pour le développement local
- Git avec Git LFS activé (`git lfs install`)

### Démarrage avec Docker

```bash
# 1. Cloner le dépôt
git clone https://github.com/AunCly/p8_confirm_mlops
cd p8_confirm_mlops

# 2. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec les valeurs souhaitées

# 3. Démarrer les services
docker compose up --build

# L'API est disponible sur [http://localhost:8000](http://localhost:8000
# La documentation interactive sur [http://localhost:8000/docs](http://localhost:8000/docs)
```

## 5. Variables d'environnement

Copier `.env.example` en `.env` et renseigner toutes les valeurs.

| Variable | Description | Exemple |
|---|---|---|
| `ENVIRONMENT` | Environnement d'exécution | `local`, `production` |
| `API_KEY` | Clé d'authentification de l'API | `<clé aléatoire sécurisée>` |

## 6. API

La documentation interactive Swagger est accessible sur [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Authentification

Tous les endpoints (sauf `/health`) requièrent l'en-tête HTTP :

```
x-api-key: <valeur de API_KEY dans .env>
```

Une clé invalide retourne `HTTP 403 Forbidden`.

## 8. Modèle de Machine Learning

- **Algorithme :** LGBMClassifier
- **Fichier :** `models/compiled/model.onnx`
- **Version sklearn :** 1.9.0

## 9. Tests

### Lancer les tests

```bash
uv run python -m pytest tests/test_api.py
```

### Configuration CI

Les tests s'exécutent automatiquement sur GitHub Actions à chaque push sur `develop` et `main` (voir section CI/CD).

## 10. CI/CD et déploiement

### Branches

| Branche | Pipeline | Action |
|---|---|---|
| `develop` | `test.yml` | Exécute les tests uniquement |
| `main` | `test_and_deploy.yml` | Exécute les tests + déploie sur Hugging Face |

### Secrets et variables GitHub Actions requis

| Nom            | Type | Description                       |
|----------------|---|-----------------------------------|
| `RENDER_TOKEN` | Secret | Token d'accès Render         |
| `RENDER_ID`    | Variable | Identifiant du Space Render |

### Déploiement sur Render Spaces

L'application est déployée sur Render Spaces via push Git. La plateforme détecte automatiquement le `Dockerfile` et construit l'image.

### Docker

#### `Dockerfile`

```dockerfile
FROM python:3.13-slim
# Installe libpq-dev et gcc pour psycopg
# Installe uv
# Copie les dépendances et le code
# Lance : fastapi run main.py --port 8000
```

#### `docker-compose.yml`

```yaml
services:
  api:          # Image FastAPI, port 8000, dépend de db
```

**Démarrage complet :**
```bash
docker compose up --build
```

**Arrêt et suppression des volumes :**
```bash
docker compose down -v
```

## 11. Notebooks d'exploration

Les notebooks Jupyter dans `notebooks/` documentent le travail de data science préalable au déploiement. Ils ne sont pas utilisés en production.

| Notebook | Objectif |
|---|---|
| `analyze.ipynb` | Exploration des données (distributions, corrélations, analyse de l'attrition) |
| `cleaning.ipynb` | Nettoyage et préparation des données brutes |
| `training.ipynb` | Entraînement du modèle, sélection des hyperparamètres, évaluation |

```bash
# Lancer Jupyter
uv run jupyter lab
```

