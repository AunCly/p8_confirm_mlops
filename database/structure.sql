-- Script de création de la table prédictions
-- SGBD : SQLite

-- Suppression si existant
DROP TABLE IF EXISTS predictions;

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sk_id_curr INTEGER NOT NULL,
    input_data TEXT NOT NULL,
    probability REAL NOT NULL,
    prediction INTEGER,
    inference_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);