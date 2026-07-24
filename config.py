"""
config.py - Central configuration for the Speech Emotion Recognition project.
All hyperparameters, paths, and settings are managed here.
"""

import os

# ─────────────────────────────────────────────
# Project Root
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# Dataset Paths
# ─────────────────────────────────────────────
DATASET_DIR      = os.path.join(BASE_DIR, "dataset")
RAVDESS_DIR      = os.path.join(DATASET_DIR, "RAVDESS")
TESS_DIR         = os.path.join(DATASET_DIR, "TESS")
EMO_DB_DIR       = os.path.join(DATASET_DIR, "EMO_DB")

# ─────────────────────────────────────────────
# Saved Artefacts
# ─────────────────────────────────────────────
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
MODELS_DIR       = os.path.join(BASE_DIR, "models")
LABELS_PATH      = os.path.join(BASE_DIR, "labels.json")

# Feature cache (speeds up repeated training runs)
FEATURES_CACHE   = os.path.join(BASE_DIR, "saved_models", "features_cache.pkl")

# ─────────────────────────────────────────────
# Audio Pre-processing
# ─────────────────────────────────────────────
SAMPLE_RATE      = 22050   # Hz – resample all audio to this rate
DURATION         = 3.0     # seconds – clip / pad each sample
N_MFCC           = 40      # number of MFCC coefficients
N_MELS           = 128     # mel-filter banks
N_CHROMA         = 12      # chroma bins
HOP_LENGTH       = 512
N_FFT            = 2048
TOP_DB           = 30      # silence-trim threshold (dB)

# ─────────────────────────────────────────────
# Emotion Labels
# ─────────────────────────────────────────────
EMOTIONS = {
    "neutral":  0,
    "calm":     1,
    "happy":    2,
    "sad":      3,
    "angry":    4,
    "fear":     5,
    "disgust":  6,
    "surprise": 7,
}
EMOTION_LABELS = list(EMOTIONS.keys())   # index → label
NUM_CLASSES    = len(EMOTIONS)

# Emoji mapping for the Streamlit UI
EMOTION_EMOJI = {
    "neutral":  "😐",
    "calm":     "😌",
    "happy":    "😊",
    "sad":      "😢",
    "angry":    "😠",
    "fear":     "😨",
    "disgust":  "🤢",
    "surprise": "😲",
}

# Colour per emotion (used in charts)
EMOTION_COLORS = {
    "neutral":  "#95a5a6",
    "calm":     "#3498db",
    "happy":    "#f1c40f",
    "sad":      "#2980b9",
    "angry":    "#e74c3c",
    "fear":     "#8e44ad",
    "disgust":  "#27ae60",
    "surprise": "#e67e22",
}

# ─────────────────────────────────────────────
# Training Hyper-parameters
# ─────────────────────────────────────────────
BATCH_SIZE        = 32
EPOCHS            = 50
LEARNING_RATE     = 0.001
VALIDATION_SPLIT  = 0.1
TEST_SPLIT        = 0.1
RANDOM_STATE      = 42

# Early-stopping / checkpoint
PATIENCE          = 10     # early-stopping patience (epochs)
MIN_DELTA         = 0.001  # minimum improvement to count as progress

# ─────────────────────────────────────────────
# Model Types
# ─────────────────────────────────────────────
MODEL_TYPES = ["cnn", "lstm", "cnn_lstm"]
DEFAULT_MODEL = "cnn_lstm"

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_FILE   = os.path.join(BASE_DIR, "ser_training.log")

# ─────────────────────────────────────────────
# Streamlit / UI
# ─────────────────────────────────────────────
APP_TITLE       = "🎙️ Speech Emotion Recognition"
APP_SUBTITLE    = "Recognize emotions from speech using Deep Learning"
MAX_UPLOAD_SIZE = 200   # MB
HISTORY_SIZE    = 50    # number of predictions kept in session history
