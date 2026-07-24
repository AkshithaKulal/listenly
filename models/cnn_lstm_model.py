"""
cnn_lstm_model.py - Hybrid CNN+LSTM model (best performance).
CNN extracts spatial features, LSTM captures temporal patterns.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    LSTM, TimeDistributed, Flatten, Dense, Dropout, Reshape, Bidirectional
)
from tensorflow.keras.regularizers import l2
import logging

logger = logging.getLogger(__name__)


def build_cnn_lstm_model(input_shape, num_classes, learning_rate=0.001):
    """
    Build a hybrid CNN+LSTM model for emotion classification.
    
    Architecture:
        Input (n_mfcc, time, 1)
        ↓
        Conv2D → BatchNorm → MaxPool → Dropout
        ↓
        Conv2D → BatchNorm → MaxPool → Dropout
        ↓
        Reshape to (time, features)
        ↓
        Bidirectional LSTM (128) → Dropout
        ↓
        Dense (128) → ReLU → Dropout
        ↓
        Dense (num_classes) → Softmax
    
    Args:
        input_shape (tuple): Shape of input (n_mfcc, time, 1)
        num_classes (int): Number of emotion classes
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        keras.Model: Compiled CNN+LSTM model
    """
    model = Sequential(name="CNN_LSTM_EmotionRecognition")
    
    # CNN Block 1
    model.add(Conv2D(
        64, (3, 3),
        padding='same',
        activation='relu',
        kernel_regularizer=l2(0.001),
        input_shape=input_shape
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # CNN Block 2
    model.add(Conv2D(
        128, (3, 3),
        padding='same',
        activation='relu',
        kernel_regularizer=l2(0.001)
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Reshape for LSTM
    # Compute the shape after CNN layers
    # We need to convert (batch, height, width, channels) → (batch, time_steps, features)
    # Here we'll flatten height×channels and keep width as time
    shape_before_lstm = model.output_shape[1:]  # (height, width, channels)
    
    # Flatten height and channels, keep width as time dimension
    # Reshape to (time_steps, features)
    new_shape = (shape_before_lstm[1], shape_before_lstm[0] * shape_before_lstm[2])
    
    model.add(Reshape(new_shape))
    
    # LSTM layers
    model.add(Bidirectional(
        LSTM(128, return_sequences=False, kernel_regularizer=l2(0.001))
    ))
    model.add(Dropout(0.3))
    
    # Dense layers
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
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
    
    logger.info(f"CNN+LSTM Model built with input shape {input_shape}")
    model.summary()
    
    return model
