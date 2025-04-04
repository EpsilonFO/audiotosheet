"""Configuration globale du projet."""

import os
from pathlib import Path

# Chemins
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MIDI_DIR = DATA_DIR / "midi"
MODELS_DIR = ROOT_DIR / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Paramètres audio
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 27.5  # A0 (la plus basse note du piano)
FMAX = 4186.0  # C8 (la plus haute note du piano)

# Paramètres du modèle
MODEL_TYPE = "cnn_lstm"  # Options: "cnn_lstm", "transformer"
DEVICE = "cuda"  # Options: "cpu", "cuda"

# Paramètres d'entraînement
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

# Paramètres d'inférence
THRESHOLD = 0.5  # Seuil de détection des notes
MIN_NOTE_DURATION = 0.05  # Durée minimale d'une note en secondes

# Paramètres de visualisation
USE_LILYPOND = True  # Utiliser Lilypond pour le rendu de partition