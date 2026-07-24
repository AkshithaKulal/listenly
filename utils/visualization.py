"""
visualization.py - Visualization utilities for audio and model metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import logging

logger = logging.getLogger(__name__)


def plot_waveform(audio, sr, title="Audio Waveform", figsize=(12, 4)):
    """
    Plot audio waveform.
    
    Args:
        audio (np.array): Audio time series
        sr (int): Sample rate
        title (str): Plot title
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    librosa.display.waveshow(audio, sr=sr, ax=ax, alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_spectrogram(audio, sr, title="Spectrogram", figsize=(12, 5)):
    """
    Plot spectrogram of audio.
    
    Args:
        audio (np.array): Audio time series
        sr (int): Sample rate
        title (str): Plot title
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='viridis')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig


def plot_mfcc(mfcc, sr, hop_length=512, title="MFCC Heatmap", figsize=(12, 5)):
    """
    Plot MFCC features as a heatmap.
    
    Args:
        mfcc (np.array): MFCC features (n_mfcc, time) or (n_mfcc, time, 1)
        sr (int): Sample rate
        hop_length (int): Hop length
        title (str): Plot title
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Remove channel dimension if present
    if len(mfcc.shape) == 3:
        mfcc = mfcc[:, :, 0]
    
    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(
        mfcc,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        ax=ax,
        cmap='coolwarm'
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("MFCC Coefficients", fontsize=12)
    fig.colorbar(img, ax=ax)
    plt.tight_layout()
    return fig


def plot_training_history(history, figsize=(14, 5)):
    """
    Plot training and validation accuracy and loss.
    
    Args:
        history (keras.callbacks.History): Training history
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names, figsize=(10, 8)):
    """
    Plot confusion matrix.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        class_names (list): List of class names
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    plt.tight_layout()
    return fig


def plot_emotion_distribution(y, class_names, title="Emotion Distribution", figsize=(10, 6)):
    """
    Plot emotion class distribution as a bar chart.
    
    Args:
        y (np.array): Label array
        class_names (list): List of class names
        title (str): Plot title
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    unique, counts = np.unique(y, return_counts=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar([class_names[i] for i in unique], counts, color='steelblue', alpha=0.8)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Emotion', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    return fig


def plot_prediction_probabilities(probs, class_names, figsize=(10, 6)):
    """
    Plot prediction probabilities as a horizontal bar chart.
    
    Args:
        probs (np.array): Probability array (num_classes,)
        class_names (list): List of class names
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by probability
    indices = np.argsort(probs)[::-1]
    sorted_probs = probs[indices]
    sorted_names = [class_names[i] for i in indices]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_probs)))
    bars = ax.barh(sorted_names, sorted_probs * 100, color=colors, alpha=0.8)
    
    ax.set_xlabel('Confidence (%)', fontsize=12)
    ax.set_title('Emotion Prediction Probabilities', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 100])
    ax.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, sorted_probs)):
        ax.text(
            prob * 100 + 1,
            bar.get_y() + bar.get_height() / 2,
            f'{prob * 100:.1f}%',
            va='center',
            fontsize=10
        )
    
    plt.tight_layout()
    return fig


def generate_classification_report_text(y_true, y_pred, class_names):
    """
    Generate classification report as a formatted string.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        class_names (list): List of class names
    
    Returns:
        str: Classification report text
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=3
    )
    return report


def save_figure(fig, filepath):
    """
    Save a matplotlib figure to file.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to save
        filepath (str): Output file path
    """
    try:
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Figure saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save figure to {filepath}: {e}")
