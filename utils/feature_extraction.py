"""
feature_extraction.py - Audio feature extraction utilities
Extracts MFCC, Mel spectrogram, Chroma, ZCR, RMS, Spectral Centroid, etc.
"""

import librosa
import numpy as np
import logging

logger = logging.getLogger(__name__)


def extract_mfcc(audio, sr, n_mfcc=40, hop_length=512, n_fft=2048):
    """
    Extract MFCC features from audio.
    
    Args:
        audio (np.array): Audio time series
        sr (int): Sample rate
        n_mfcc (int): Number of MFCC coefficients
        hop_length (int): Hop length for STFT
        n_fft (int): FFT window size
    
    Returns:
        np.array: MFCC features (n_mfcc, time)
    """
    try:
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=n_mfcc,
            hop_length=hop_length,
            n_fft=n_fft
        )
        return mfcc
    except Exception as e:
        logger.error(f"Failed to extract MFCC: {e}")
        raise


def extract_mel_spectrogram(audio, sr, n_mels=128, hop_length=512, n_fft=2048):
    """
    Extract Mel spectrogram from audio.
    
    Args:
        audio (np.array): Audio time series
        sr (int): Sample rate
        n_mels (int): Number of mel bands
        hop_length (int): Hop length for STFT
        n_fft (int): FFT window size
    
    Returns:
        np.array: Mel spectrogram (n_mels, time)
    """
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft
        )
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    except Exception as e:
        logger.error(f"Failed to extract Mel spectrogram: {e}")
        raise


def extract_chroma(audio, sr, hop_length=512, n_chroma=12):
    """
    Extract Chroma features from audio.
    
    Args:
        audio (np.array): Audio time series
        sr (int): Sample rate
        hop_length (int): Hop length
        n_chroma (int): Number of chroma bins
    
    Returns:
        np.array: Chroma features (n_chroma, time)
    """
    try:
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=sr,
            hop_length=hop_length,
            n_chroma=n_chroma
        )
        return chroma
    except Exception as e:
        logger.error(f"Failed to extract Chroma: {e}")
        raise


def extract_zcr(audio, hop_length=512):
    """
    Extract Zero Crossing Rate from audio.
    
    Args:
        audio (np.array): Audio time series
        hop_length (int): Hop length
    
    Returns:
        np.array: ZCR (1, time)
    """
    try:
        zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=hop_length)
        return zcr
    except Exception as e:
        logger.error(f"Failed to extract ZCR: {e}")
        raise


def extract_rms(audio, hop_length=512):
    """
    Extract RMS Energy from audio.
    
    Args:
        audio (np.array): Audio time series
        hop_length (int): Hop length
    
    Returns:
        np.array: RMS energy (1, time)
    """
    try:
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)
        return rms
    except Exception as e:
        logger.error(f"Failed to extract RMS: {e}")
        raise


def extract_spectral_centroid(audio, sr, hop_length=512, n_fft=2048):
    """
    Extract Spectral Centroid from audio.
    
    Args:
        audio (np.array): Audio time series
        sr (int): Sample rate
        hop_length (int): Hop length
        n_fft (int): FFT window size
    
    Returns:
        np.array: Spectral centroid (1, time)
    """
    try:
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=sr,
            hop_length=hop_length,
            n_fft=n_fft
        )
        return centroid
    except Exception as e:
        logger.error(f"Failed to extract Spectral Centroid: {e}")
        raise


def extract_all_features(audio, sr, n_mfcc=40, n_mels=128, n_chroma=12, hop_length=512, n_fft=2048):
    """
    Extract all audio features and concatenate them.
    
    Features extracted:
        - MFCC
        - Mel Spectrogram
        - Chroma
        - Zero Crossing Rate
        - RMS Energy
        - Spectral Centroid
    
    Args:
        audio (np.array): Audio time series
        sr (int): Sample rate
        n_mfcc (int): Number of MFCC coefficients
        n_mels (int): Number of mel bands
        n_chroma (int): Number of chroma bins
        hop_length (int): Hop length
        n_fft (int): FFT window size
    
    Returns:
        np.array: Concatenated feature vector (features, time)
    """
    try:
        # Extract individual features
        mfcc = extract_mfcc(audio, sr, n_mfcc, hop_length, n_fft)
        mel_spec = extract_mel_spectrogram(audio, sr, n_mels, hop_length, n_fft)
        chroma = extract_chroma(audio, sr, hop_length, n_chroma)
        zcr = extract_zcr(audio, hop_length)
        rms = extract_rms(audio, hop_length)
        centroid = extract_spectral_centroid(audio, sr, hop_length, n_fft)
        
        # Ensure all features have the same time dimension
        min_time = min(
            mfcc.shape[1],
            mel_spec.shape[1],
            chroma.shape[1],
            zcr.shape[1],
            rms.shape[1],
            centroid.shape[1]
        )
        
        mfcc = mfcc[:, :min_time]
        mel_spec = mel_spec[:, :min_time]
        chroma = chroma[:, :min_time]
        zcr = zcr[:, :min_time]
        rms = rms[:, :min_time]
        centroid = centroid[:, :min_time]
        
        # Concatenate all features
        features = np.vstack([mfcc, mel_spec, chroma, zcr, rms, centroid])
        
        logger.debug(f"Extracted features shape: {features.shape}")
        return features
        
    except Exception as e:
        logger.error(f"Failed to extract all features: {e}")
        raise


def compute_statistics(features):
    """
    Compute statistical aggregations over time dimension.
    Returns mean and std of each feature.
    
    Args:
        features (np.array): Feature matrix (features, time)
    
    Returns:
        np.array: Flattened feature vector [mean, std]
    """
    try:
        mean = np.mean(features, axis=1)
        std = np.std(features, axis=1)
        feature_vector = np.concatenate([mean, std])
        return feature_vector
    except Exception as e:
        logger.error(f"Failed to compute statistics: {e}")
        raise


def extract_features_for_cnn(audio, sr, n_mfcc=40, hop_length=512, n_fft=2048, use_deltas=True):
    """
    Extract MFCC features suitable for CNN input (2D image-like).

    When use_deltas=True, stacks MFCC + delta + delta-delta as channels:
    shape (n_mfcc, time, 3). Otherwise shape is (n_mfcc, time, 1).
    """
    try:
        mfcc = extract_mfcc(audio, sr, n_mfcc, hop_length, n_fft)
        if use_deltas:
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            features = np.stack([mfcc, delta, delta2], axis=-1).astype(np.float32)
        else:
            features = np.expand_dims(mfcc, axis=-1).astype(np.float32)
        return features
    except Exception as e:
        logger.error(f"Failed to extract CNN features: {e}")
        raise


def extract_features_for_lstm(audio, sr, n_mfcc=40, hop_length=512, n_fft=2048, use_deltas=True):
    """
    Extract MFCC features suitable for LSTM input (time series).

    When use_deltas=True, concatenates MFCC + delta + delta-delta along
    the feature axis: shape (time, n_mfcc * 3).
    """
    try:
        mfcc = extract_mfcc(audio, sr, n_mfcc, hop_length, n_fft)
        if use_deltas:
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            features = np.concatenate([mfcc, delta, delta2], axis=0).T.astype(np.float32)
        else:
            features = mfcc.T.astype(np.float32)
        return features
    except Exception as e:
        logger.error(f"Failed to extract LSTM features: {e}")
        raise
