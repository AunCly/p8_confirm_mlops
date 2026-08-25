import datetime
import json
import sqlite3

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

def get_engine():
    database_path = Path(__file__).parent / "database.sqlite"
    engine = sqlite3.connect(database_path)
    return engine

def create_database():

    print('Creating database...')
    conn = get_engine()

    # Si le fichier de base de données n'existe pas, créer la structure de la base de données
    structure_sql = Path(__file__).parent / "structure.sql"

    if structure_sql.exists():
        conn.execute("PRAGMA foreign_keys = ON;")  # Activer les clés étrangères pour SQLite
        try:
            with open(structure_sql, "r") as f:
                sql_script = f.read()
                conn.executescript(sql_script)

        except FileNotFoundError:
            print(f"Erreur : Le fichier '{structure_sql}' est introuvable.")
        except Exception as e:
            print(f"Une erreur est survenue lors de l'exécution de la base de données : {e}")

def save_prediction(sk_id_curr, input_data, prediction, probability, inference_time):
    engine = get_engine()
    with engine:
        engine.execute(
            "INSERT INTO predictions (sk_id_curr, input_data, prediction, probability, inference_time, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                sk_id_curr,
                json.dumps(input_data),
                prediction,
                probability,
                inference_time,
                str(datetime.datetime.now())
            ),
        )


def get_predictions():
    engine = get_engine()
    with engine:
        result = engine.execute("SELECT * FROM predictions")

        colonnes = [col[0] for col in result.description]

        predictions = []
        for row in result.fetchall():
            row_dict = dict(zip(colonnes, row))

            if 'input_data' in row_dict and isinstance(row_dict['input_data'], str):
                row_dict['input_data'] = json.loads(row_dict['input_data'])

            predictions.append(row_dict)

        return predictions
