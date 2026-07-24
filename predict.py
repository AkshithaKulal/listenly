"""
predict.py - Prediction script for Speech Emotion Recognition.
Loads a trained model and predicts emotion from audio file.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from tensorflow.keras.models import load_model

import config
from utils.audio_processing import preprocess_audio
from utils.feature_extraction import extract_features_for_cnn, extract_features_for_lstm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class EmotionPredictor:
    """
    Emotion predictor class that loads model and makes predictions.
    """
    
    def __init__(self, model_path, labels_path=None, model_type="cnn_lstm"):
        """
        Initialize predictor.
        
        Args:
            model_path (str): Path to trained model (.h5 file)
            labels_path (str): Path to labels JSON file
            model_type (str): Type of model ('cnn', 'lstm', 'cnn_lstm')
        """
        self.model_type = model_type
        self.model = self._load_model(model_path)
        self.labels = self._load_labels(labels_path)
        logger.info(f"Predictor initialized with model: {model_path}")
    
    def _load_model(self, model_path):
        """Load trained Keras model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            model = load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_labels(self, labels_path):
        """Load emotion labels from JSON."""
        if labels_path is None:
            labels_path = config.LABELS_PATH
        
        if not os.path.exists(labels_path):
            logger.warning(f"Labels file not found: {labels_path}. Using default.")
            return config.EMOTION_LABELS
        
        try:
            with open(labels_path, 'r') as f:
                labels = json.load(f)
            logger.info(f"Labels loaded: {labels}")
            return labels
        except Exception as e:
            logger.warning(f"Failed to load labels: {e}. Using default.")
            return config.EMOTION_LABELS
    
    def extract_features(self, audio_path):
        """
        Extract features from audio file.
        
        Args:
            audio_path (str): Path to audio file
        
        Returns:
            np.array: Extracted features ready for model input
        """
        # Preprocess audio
        audio, sr = preprocess_audio(
            audio_path,
            sr=config.SAMPLE_RATE,
            duration=config.DURATION,
            top_db=config.TOP_DB,
            apply_noise_reduction=True
        )
        
        # Extract features based on model type
        if self.model_type in ("cnn", "cnn_lstm"):
            features = extract_features_for_cnn(
                audio, sr,
                n_mfcc=config.N_MFCC,
                hop_length=config.HOP_LENGTH,
                n_fft=config.N_FFT
            )
        else:  # lstm
            features = extract_features_for_lstm(
                audio, sr,
                n_mfcc=config.N_MFCC,
                hop_length=config.HOP_LENGTH,
                n_fft=config.N_FFT
            )
        
        # Add batch dimension
        features = np.expand_dims(features, axis=0)
        return features
    
    def predict(self, audio_path):
        """
        Predict emotion from audio file.
        
        Args:
            audio_path (str): Path to audio file (.wav)
        
        Returns:
            dict: Prediction results containing:
                - emotion: predicted emotion label
                - confidence: confidence score (0-1)
                - probabilities: dict of all emotion probabilities
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Processing: {audio_path}")
        
        # Extract features
        features = self.extract_features(audio_path)
        
        # Predict
        predictions = self.model.predict(features, verbose=0)[0]
        
        # Get predicted class
        predicted_idx = np.argmax(predictions)
        predicted_emotion = self.labels[predicted_idx]
        confidence = predictions[predicted_idx]
        
        # Create probability dictionary
        probabilities = {
            self.labels[i]: float(predictions[i])
            for i in range(len(self.labels))
        }
        
        result = {
            "emotion": predicted_emotion,
            "confidence": float(confidence),
            "probabilities": probabilities
        }
        
        logger.info(f"Predicted: {predicted_emotion} (confidence: {confidence:.2%})")
        return result
    
    def predict_batch(self, audio_paths):
        """
        Predict emotions for multiple audio files.
        
        Args:
            audio_paths (list): List of audio file paths
        
        Returns:
            list: List of prediction results
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path)
                result["file"] = audio_path
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict {audio_path}: {e}")
                results.append({
                    "file": audio_path,
                    "error": str(e)
                })
        return results


def print_prediction(result):
    """Pretty print prediction result."""
    print("\n" + "=" * 60)
    print(f"🎙️  Emotion Prediction Result")
    print("=" * 60)
    print(f"Predicted Emotion: {result['emotion'].upper()} {config.EMOTION_EMOJI.get(result['emotion'], '')}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nProbability Distribution:")
    print("-" * 60)
    
    # Sort by probability
    sorted_probs = sorted(
        result['probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for emotion, prob in sorted_probs:
        emoji = config.EMOTION_EMOJI.get(emotion, "")
        bar = "█" * int(prob * 50)
        print(f"{emotion:10s} {emoji} │{bar:<50}│ {prob:.2%}")
    
    print("=" * 60 + "\n")


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description="Predict emotion from speech audio file"
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to audio file (.wav)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (.h5). If not specified, uses best_model_cnn_lstm.h5"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="cnn_lstm",
        choices=["cnn", "lstm", "cnn_lstm"],
        help="Type of model architecture"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Path to labels JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file to save prediction results"
    )
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model is None:
        model_path = os.path.join(
            config.SAVED_MODELS_DIR,
            f"best_model_{args.model_type}.h5"
        )
        if not os.path.exists(model_path):
            model_path = os.path.join(
                config.SAVED_MODELS_DIR,
                f"final_model_{args.model_type}.h5"
            )
    else:
        model_path = args.model
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.error("Please train a model first using train.py")
        return 1
    
    try:
        # Initialize predictor
        predictor = EmotionPredictor(
            model_path=model_path,
            labels_path=args.labels,
            model_type=args.model_type
        )
        
        # Make prediction
        result = predictor.predict(args.audio_file)
        
        # Print result
        print_prediction(result)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Prediction saved to {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
