"""
utils/__init__.py - Utilities package for Speech Emotion Recognition
"""

from .audio_processing import (
    preprocess_audio,
    remove_silence,
    normalize_audio,
    reduce_noise,
    resample_audio,
    pad_or_truncate,
    augment_audio
)

from .feature_extraction import (
    extract_mfcc,
    extract_mel_spectrogram,
    extract_chroma,
    extract_zcr,
    extract_rms,
    extract_spectral_centroid,
    extract_all_features,
    extract_features_for_cnn,
    extract_features_for_lstm,
    compute_statistics
)

from .dataset_loader import (
    load_all_datasets,
    prepare_dataset,
    load_ravdess,
    load_tess,
    load_emodb
)

from .visualization import (
    plot_waveform,
    plot_spectrogram,
    plot_mfcc,
    plot_training_history,
    plot_confusion_matrix,
    plot_emotion_distribution,
    plot_prediction_probabilities,
    generate_classification_report_text,
    save_figure
)

__all__ = [
    # audio_processing
    "preprocess_audio",
    "remove_silence",
    "normalize_audio",
    "reduce_noise",
    "resample_audio",
    "pad_or_truncate",
    "augment_audio",
    # feature_extraction
    "extract_mfcc",
    "extract_mel_spectrogram",
    "extract_chroma",
    "extract_zcr",
    "extract_rms",
    "extract_spectral_centroid",
    "extract_all_features",
    "extract_features_for_cnn",
    "extract_features_for_lstm",
    "compute_statistics",
    # dataset_loader
    "load_all_datasets",
    "prepare_dataset",
    "load_ravdess",
    "load_tess",
    "load_emodb",
    # visualization
    "plot_waveform",
    "plot_spectrogram",
    "plot_mfcc",
    "plot_training_history",
    "plot_confusion_matrix",
    "plot_emotion_distribution",
    "plot_prediction_probabilities",
    "generate_classification_report_text",
    "save_figure",
]
