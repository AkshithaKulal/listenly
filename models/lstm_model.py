"""
lstm_model.py - LSTM model for emotion recognition from sequential MFCC features.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.regularizers import l2
import logging

logger = logging.getLogger(__name__)


def build_lstm_model(input_shape, num_classes, learning_rate=0.001):
    """
    Build an LSTM model for emotion classification.
    
    Architecture:
        Input (time_steps, features)
        ↓
        Bidirectional LSTM (128) → Dropout
        ↓
        Bidirectional LSTM (64) → Dropout
        ↓
        Dense (128) → ReLU → BatchNorm → Dropout
        ↓
        Dense (num_classes) → Softmax
    
    Args:
        input_shape (tuple): Shape of input (time_steps, n_mfcc)
        num_classes (int): Number of emotion classes
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        keras.Model: Compiled LSTM model
    """
    model = Sequential(name="LSTM_EmotionRecognition")
    
    # First Bidirectional LSTM layer
    model.add(Bidirectional(
        LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)),
        input_shape=input_shape
    ))
    model.add(Dropout(0.3))
    
    # Second Bidirectional LSTM layer
    model.add(Bidirectional(
        LSTM(64, return_sequences=False, kernel_regularizer=l2(0.001))
    ))
    model.add(Dropout(0.3))
    
    # Dense layers
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"LSTM Model built with input shape {input_shape}")
    model.summary()
    
    return model
