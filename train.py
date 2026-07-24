"""
train.py - Main training script for Speech Emotion Recognition.
Loads datasets, extracts features, trains model, and evaluates performance.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Local imports
import config
from utils.dataset_loader import load_all_datasets, prepare_dataset
from utils.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_emotion_distribution,
    generate_classification_report_text,
    save_figure
)
from models import build_cnn_model, build_lstm_model, build_cnn_lstm_model

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    logger.info("Directories verified")


def save_labels():
    """Save emotion labels mapping to JSON file."""
    with open(config.LABELS_PATH, 'w') as f:
        json.dump(config.EMOTION_LABELS, f, indent=2)
    logger.info(f"Labels saved to {config.LABELS_PATH}")


def get_model_builder(model_type):
    """Get model builder function based on type."""
    builders = {
        "cnn": build_cnn_model,
        "lstm": build_lstm_model,
        "cnn_lstm": build_cnn_lstm_model,
    }
    if model_type not in builders:
        raise ValueError(f"Invalid model type: {model_type}. Choose from {list(builders.keys())}")
    return builders[model_type]


def train_model(model_type="cnn_lstm", use_cache=True):
    """
    Main training pipeline.
    
    Args:
        model_type (str): Model architecture ('cnn', 'lstm', or 'cnn_lstm')
        use_cache (bool): Use cached features if available
    
    Returns:
        tuple: (model, history, test_data)
    """
    logger.info("=" * 80)
    logger.info(f"Starting training with model type: {model_type}")
    logger.info("=" * 80)
    
    # 1. Setup
    setup_directories()
    save_labels()
    
    # 2. Load datasets
    logger.info("Loading datasets...")
    df = load_all_datasets()
    logger.info(f"Total samples: {len(df)}")
    
    # 3. Extract features
    logger.info("Extracting features...")
    X, y = prepare_dataset(df, model_type=model_type, use_cache=use_cache)
    logger.info(f"Feature shape: {X.shape}, Labels shape: {y.shape}")
    
    # Visualize emotion distribution
    fig_dist = plot_emotion_distribution(y, config.EMOTION_LABELS, title="Training Data Distribution")
    save_figure(fig_dist, os.path.join(config.SAVED_MODELS_DIR, "emotion_distribution.png"))
    plt.close(fig_dist)
    
    # 4. Split data
    logger.info("Splitting dataset...")
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = config.VALIDATION_SPLIT / (1 - config.TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=config.RANDOM_STATE,
        stratify=y_temp
    )
    
    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # 5. Build model
    logger.info(f"Building {model_type.upper()} model...")
    model_builder = get_model_builder(model_type)
    model = model_builder(
        input_shape=X_train.shape[1:],
        num_classes=config.NUM_CLASSES,
        learning_rate=config.LEARNING_RATE
    )
    
    # 6. Setup callbacks
    model_path = os.path.join(config.SAVED_MODELS_DIR, f"best_model_{model_type}.h5")
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # 7. Train model
    logger.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Training completed!")
    
    # 8. Save final model
    final_model_path = os.path.join(config.SAVED_MODELS_DIR, f"final_model_{model_type}.h5")
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # 9. Plot training history
    fig_history = plot_training_history(history)
    save_figure(fig_history, os.path.join(config.SAVED_MODELS_DIR, f"training_history_{model_type}.png"))
    plt.close(fig_history)
    
    # 10. Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Confusion matrix
    fig_cm = plot_confusion_matrix(y_test, y_pred, config.EMOTION_LABELS)
    save_figure(fig_cm, os.path.join(config.SAVED_MODELS_DIR, f"confusion_matrix_{model_type}.png"))
    plt.close(fig_cm)
    
    # Classification report
    report = generate_classification_report_text(y_test, y_pred, config.EMOTION_LABELS)
    logger.info("\nClassification Report:\n" + report)
    
    # Save report to file
    report_path = os.path.join(config.SAVED_MODELS_DIR, f"classification_report_{model_type}.txt")
    with open(report_path, 'w') as f:
        f.write(f"Model: {model_type}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write(report)
    logger.info(f"Classification report saved to {report_path}")
    
    # Save training metrics
    metrics = {
        "model_type": model_type,
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "epochs_trained": len(history.history['loss']),
        "best_val_accuracy": float(max(history.history['val_accuracy'])),
        "best_val_loss": float(min(history.history['val_loss'])),
    }
    
    metrics_path = os.path.join(config.SAVED_MODELS_DIR, f"metrics_{model_type}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    logger.info("=" * 80)
    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 80)
    
    return model, history, (X_test, y_test, y_pred)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Train Speech Emotion Recognition Model")
    parser.add_argument(
        "--model",
        type=str,
        default=config.DEFAULT_MODEL,
        choices=config.MODEL_TYPES,
        help="Model type to train"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached features (re-extract)"
    )
    
    args = parser.parse_args()
    
    try:
        model, history, test_data = train_model(
            model_type=args.model,
            use_cache=not args.no_cache
        )
        logger.info("Training completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
