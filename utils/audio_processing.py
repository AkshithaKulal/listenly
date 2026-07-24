"""
audio_processing.py - Audio preprocessing utilities
Handles silence removal, noise reduction, normalization, and resampling.
"""

import librosa
import numpy as np
import logging

logger = logging.getLogger(__name__)


def remove_silence(audio, sr, top_db=30):
    """
    Remove leading and trailing silence from audio signal.
    
    Args:
        audio (np.array): Audio time series
        sr (int): Sample rate
        top_db (int): Threshold for silence (in dB)
    
    Returns:
        np.array: Audio with silence trimmed
    """
    try:
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
    except Exception as e:
        logger.warning(f"Failed to trim silence: {e}. Returning original audio.")
        return audio


def normalize_audio(audio):
    """
    Normalize audio to [-1, 1] range.
    
    Args:
        audio (np.array): Audio time series
    
    Returns:
        np.array: Normalized audio
    """
    try:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    except Exception as e:
        logger.warning(f"Failed to normalize audio: {e}. Returning original.")
        return audio


def reduce_noise(audio, sr, n_std_thresh=1.5):
    """
    Simple noise reduction using spectral gating approach.
    
    Args:
        audio (np.array): Audio time series
        sr (int): Sample rate
        n_std_thresh (float): Threshold for noise gate
    
    Returns:
        np.array: Noise-reduced audio
    """
    try:
        # Compute short-time Fourier transform
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Estimate noise from quietest frames
        noise_profile = np.median(magnitude, axis=1, keepdims=True)
        
        # Apply spectral gate
        mask = magnitude > (noise_profile * n_std_thresh)
        stft_denoised = stft * mask
        
        # Inverse STFT
        audio_denoised = librosa.istft(stft_denoised)
        return audio_denoised
    except Exception as e:
        logger.warning(f"Failed to reduce noise: {e}. Returning original audio.")
        return audio


def resample_audio(audio, orig_sr, target_sr):
    """
    Resample audio to target sample rate.
    
    Args:
        audio (np.array): Audio time series
        orig_sr (int): Original sample rate
        target_sr (int): Target sample rate
    
    Returns:
        np.array: Resampled audio
    """
    try:
        if orig_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        return audio
    except Exception as e:
        logger.error(f"Failed to resample audio: {e}")
        raise


def pad_or_truncate(audio, sr, duration):
    """
    Pad or truncate audio to a fixed duration.
    
    Args:
        audio (np.array): Audio time series
        sr (int): Sample rate
        duration (float): Target duration in seconds
    
    Returns:
        np.array: Fixed-length audio
    """
    try:
        max_len = int(sr * duration)
        if len(audio) < max_len:
            # Pad with zeros
            pad_width = max_len - len(audio)
            audio = np.pad(audio, (0, pad_width), mode='constant')
        else:
            # Truncate
            audio = audio[:max_len]
        return audio
    except Exception as e:
        logger.error(f"Failed to pad/truncate audio: {e}")
        raise


def preprocess_audio(audio_path, sr=22050, duration=3.0, top_db=30, apply_noise_reduction=True):
    """
    Full preprocessing pipeline for audio.
    
    Steps:
        1. Load audio
        2. Resample to target sample rate
        3. Remove silence
        4. Reduce noise (optional)
        5. Normalize
        6. Pad or truncate to fixed duration
    
    Args:
        audio_path (str): Path to audio file
        sr (int): Target sample rate
        duration (float): Target duration in seconds
        top_db (int): Threshold for silence removal
        apply_noise_reduction (bool): Whether to apply noise reduction
    
    Returns:
        np.array: Preprocessed audio
        int: Sample rate
    """
    try:
        # Load audio
        audio, orig_sr = librosa.load(audio_path, sr=None)
        logger.debug(f"Loaded audio from {audio_path}: {len(audio)} samples @ {orig_sr} Hz")
        
        # Resample
        audio = resample_audio(audio, orig_sr, sr)
        
        # Remove silence
        audio = remove_silence(audio, sr, top_db)
        
        # Reduce noise
        if apply_noise_reduction:
            audio = reduce_noise(audio, sr)
        
        # Normalize
        audio = normalize_audio(audio)
        
        # Pad or truncate
        audio = pad_or_truncate(audio, sr, duration)
        
        logger.debug(f"Preprocessed audio: {len(audio)} samples @ {sr} Hz")
        return audio, sr
        
    except Exception as e:
        logger.error(f"Error preprocessing {audio_path}: {e}")
        raise


def augment_audio(audio, sr):
    """
    Apply data augmentation to audio (time stretch, pitch shift, noise).
    
    Args:
        audio (np.array): Audio time series
        sr (int): Sample rate
    
    Returns:
        list: List of augmented audio samples
    """
    augmented = []
    
    try:
        # Original
        augmented.append(audio)
        
        # Time stretch
        audio_stretched = librosa.effects.time_stretch(audio, rate=0.9)
        audio_stretched = pad_or_truncate(audio_stretched, sr, len(audio) / sr)
        augmented.append(audio_stretched)
        
        # Pitch shift
        audio_pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
        augmented.append(audio_pitched)
        
        # Add noise
        noise = np.random.randn(len(audio)) * 0.005
        audio_noisy = audio + noise
        audio_noisy = normalize_audio(audio_noisy)
        augmented.append(audio_noisy)
        
    except Exception as e:
        logger.warning(f"Audio augmentation failed: {e}. Returning original only.")
        return [audio]
    
    return augmented
