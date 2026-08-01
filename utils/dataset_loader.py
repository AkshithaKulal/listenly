"""
dataset_loader.py - Dataset loading utilities
Supports RAVDESS, TESS, and EMO-DB datasets.
Automatically detects which datasets are present.
"""

import os
import re
import glob
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.audio_processing import preprocess_audio
from utils.feature_extraction import extract_features_for_cnn, extract_features_for_lstm
import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# RAVDESS Loader
# ─────────────────────────────────────────────
# Filename format: 03-01-06-01-02-01-12.wav
# Modality-Vocal-Emotion-Intensity-Statement-Repetition-Actor
# Emotion codes: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised

RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}


def load_ravdess(ravdess_dir):
    """
    Load RAVDESS dataset audio files and extract labels.
    
    Supports both flat and per-actor subdirectory structure.
    
    Args:
        ravdess_dir (str): Path to RAVDESS dataset directory
    
    Returns:
        list: List of (filepath, emotion_label) tuples
    """
    data = []
    # Search recursively for wav files
    pattern = os.path.join(ravdess_dir, "**", "*.wav")
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        logger.warning(f"No .wav files found in {ravdess_dir}")
        return data
    
    logger.info(f"Found {len(files)} RAVDESS files")
    
    for filepath in files:
        try:
            filename = os.path.basename(filepath)
            parts = filename.replace(".wav", "").split("-")
            
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = RAVDESS_EMOTION_MAP.get(emotion_code)
                if emotion and emotion in config.EMOTIONS:
                    data.append((filepath, emotion))
        except Exception as e:
            logger.warning(f"Failed to parse RAVDESS filename {filepath}: {e}")
    
    logger.info(f"Loaded {len(data)} valid RAVDESS samples")
    return data


# ─────────────────────────────────────────────
# TESS Loader
# ─────────────────────────────────────────────
# Structure: TESS/<YOW_emotion>/<filename>.wav  or  TESS/<filename>.wav
# Emotion is in the folder name or file name (e.g., OAF_happy, YAF_angry)

TESS_EMOTION_MAP = {
    "angry":    "angry",
    "disgust":  "disgust",
    "fear":     "fear",
    "happy":    "happy",
    "neutral":  "neutral",
    "ps":       "surprise",   # TESS uses "ps" for "pleasant surprise"
    "sad":      "sad",
}


def load_tess(tess_dir):
    """
    Load TESS dataset audio files and extract labels.
    
    Args:
        tess_dir (str): Path to TESS dataset directory
    
    Returns:
        list: List of (filepath, emotion_label) tuples
    """
    data = []
    pattern = os.path.join(tess_dir, "**", "*.wav")
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        logger.warning(f"No .wav files found in {tess_dir}")
        return data
    
    logger.info(f"Found {len(files)} TESS files")
    
    for filepath in files:
        try:
            filename = os.path.basename(filepath).replace(".wav", "").lower()
            
            # Try folder-based label first
            parent_dir = os.path.basename(os.path.dirname(filepath)).lower()
            
            emotion = None
            # Search emotion keyword in filename and parent folder
            for key, label in TESS_EMOTION_MAP.items():
                if key in filename or key in parent_dir:
                    emotion = label
                    break
            
            if emotion and emotion in config.EMOTIONS:
                data.append((filepath, emotion))
        except Exception as e:
            logger.warning(f"Failed to parse TESS filename {filepath}: {e}")
    
    logger.info(f"Loaded {len(data)} valid TESS samples")
    return data


# ─────────────────────────────────────────────
# EMO-DB Loader
# ─────────────────────────────────────────────
# Filename format: 03a01Fa.wav
# Character at index 5 is the emotion code
# W=anger, L=boredom, E=disgust, A=anxiety/fear, F=happiness, T=sadness, N=neutral

EMODB_EMOTION_MAP = {
    "W": "angry",
    "L": "neutral",   # boredom → neutral
    "E": "disgust",
    "A": "fear",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}


def load_emodb(emodb_dir):
    """
    Load EMO-DB dataset audio files and extract labels.
    
    Args:
        emodb_dir (str): Path to EMO-DB dataset directory
    
    Returns:
        list: List of (filepath, emotion_label) tuples
    """
    data = []
    pattern = os.path.join(emodb_dir, "**", "*.wav")
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        logger.warning(f"No .wav files found in {emodb_dir}")
        return data
    
    logger.info(f"Found {len(files)} EMO-DB files")
    
    for filepath in files:
        try:
            filename = os.path.basename(filepath)
            if len(filename) >= 6:
                emotion_code = filename[5].upper()
                emotion = EMODB_EMOTION_MAP.get(emotion_code)
                if emotion and emotion in config.EMOTIONS:
                    data.append((filepath, emotion))
        except Exception as e:
            logger.warning(f"Failed to parse EMO-DB filename {filepath}: {e}")
    
    logger.info(f"Loaded {len(data)} valid EMO-DB samples")
    return data


# ─────────────────────────────────────────────
# Unified Dataset Loader
# ─────────────────────────────────────────────

def load_all_datasets():
    """
    Automatically detect and load all available datasets.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['filepath', 'emotion', 'source']
    """
    all_data = []
    
    datasets = [
        ("RAVDESS", config.RAVDESS_DIR, load_ravdess),
        ("TESS",    config.TESS_DIR,    load_tess),
        ("EMO-DB",  config.EMO_DB_DIR,  load_emodb),
    ]
    
    for name, path, loader_fn in datasets:
        if os.path.exists(path) and any(
            glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)
        ):
            logger.info(f"Loading {name} dataset from {path}...")
            try:
                samples = loader_fn(path)
                for filepath, emotion in samples:
                    all_data.append({
                        "filepath": filepath,
                        "emotion":  emotion,
                        "source":   name,
                    })
                logger.info(f"{name}: {len(samples)} samples loaded")
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
        else:
            logger.warning(f"{name} dataset not found at {path}. Skipping.")
    
    if not all_data:
        raise RuntimeError(
            "No datasets found! Place audio files in dataset/RAVDESS, dataset/TESS, or dataset/EMO_DB."
        )
    
    df = pd.DataFrame(all_data)
    logger.info(f"Total samples loaded: {len(df)}")
    logger.info(f"Emotion distribution:\n{df['emotion'].value_counts()}")
    return df


# ─────────────────────────────────────────────
# Feature Extraction Pipeline
# ─────────────────────────────────────────────

def prepare_dataset(df, model_type="cnn_lstm", use_cache=True, augment=None):
    """
    Process all audio files and extract features.

    Args:
        df (pd.DataFrame): DataFrame with 'filepath' and 'emotion' columns
        model_type (str): One of 'cnn', 'lstm', 'cnn_lstm'
        use_cache (bool): Load from cache if available
        augment (bool|None): Override config.USE_AUGMENTATION when set

    Returns:
        tuple: (X, y) numpy arrays
    """
    import pickle
    from utils.audio_processing import augment_audio

    use_aug = config.USE_AUGMENTATION if augment is None else augment
    use_deltas = getattr(config, "USE_MFCC_DELTAS", True)
    cache_tag = f"{model_type}_d{int(use_deltas)}_a{int(use_aug)}"
    cache_path = config.FEATURES_CACHE.replace(".pkl", f"_{cache_tag}.pkl")

    if use_cache and os.path.exists(cache_path):
        logger.info(f"Loading features from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    X_list = []
    y_list = []
    failed = 0

    logger.info(
        f"Extracting features for {len(df)} audio files "
        f"(deltas={use_deltas}, augment={use_aug})..."
    )

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        try:
            audio, sr = preprocess_audio(
                row["filepath"],
                sr=config.SAMPLE_RATE,
                duration=config.DURATION,
                top_db=config.TOP_DB,
                apply_noise_reduction=False,
            )

            audio_versions = (
                augment_audio(audio, sr, factor=getattr(config, "AUGMENT_FACTOR", 3))
                if use_aug
                else [audio]
            )

            emotion_label = config.EMOTIONS[row["emotion"]]

            for sample in audio_versions:
                if model_type in ("cnn", "cnn_lstm"):
                    features = extract_features_for_cnn(
                        sample,
                        sr,
                        n_mfcc=config.N_MFCC,
                        hop_length=config.HOP_LENGTH,
                        n_fft=config.N_FFT,
                        use_deltas=use_deltas,
                    )
                else:
                    features = extract_features_for_lstm(
                        sample,
                        sr,
                        n_mfcc=config.N_MFCC,
                        hop_length=config.HOP_LENGTH,
                        n_fft=config.N_FFT,
                        use_deltas=use_deltas,
                    )
                X_list.append(features)
                y_list.append(emotion_label)

        except Exception as e:
            logger.warning(f"Skipping {row['filepath']}: {e}")
            failed += 1

    if failed > 0:
        logger.warning(f"Failed to process {failed} files")

    if not X_list:
        raise RuntimeError("No features were successfully extracted!")

    target_time = _get_target_time(X_list, model_type)
    X_list = [_fix_time_dim(feat, target_time, model_type) for feat in X_list]

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    logger.info(f"Feature array shape: {X.shape}, Labels shape: {y.shape}")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((X, y), f)
    logger.info(f"Features cached to {cache_path}")

    return X, y


def _get_target_time(feature_list, model_type):
    """Compute the median time dimension across all samples."""
    if model_type in ("cnn", "cnn_lstm"):
        times = [f.shape[1] for f in feature_list]  # (n_mfcc, time, channels)
    else:
        times = [f.shape[0] for f in feature_list]  # (time, features)
    return int(np.median(times))


def _fix_time_dim(features, target_time, model_type):
    """Pad or truncate a single sample's time dimension to target_time."""
    if model_type in ("cnn", "cnn_lstm"):
        current_time = features.shape[1]  # (n_mfcc, time, channels)
        channels = features.shape[2] if features.ndim == 3 else 1
        if features.ndim == 2:
            features = np.expand_dims(features, axis=-1)
            channels = 1
        if current_time < target_time:
            pad = np.zeros(
                (features.shape[0], target_time - current_time, channels),
                dtype=np.float32,
            )
            features = np.concatenate([features, pad], axis=1)
        else:
            features = features[:, :target_time, :]
    else:  # lstm
        current_time = features.shape[0]  # (time, features)
        if current_time < target_time:
            pad = np.zeros(
                (target_time - current_time, features.shape[1]),
                dtype=np.float32,
            )
            features = np.concatenate([features, pad], axis=0)
        else:
            features = features[:target_time, :]
    return features
