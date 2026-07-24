"""
cnn_model.py - Convolutional Neural Network for emotion recognition.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout, Activation
)
from tensorflow.keras.regularizers import l2
import logging

logger = logging.getLogger(__name__)


def build_cnn_model(input_shape, num_classes, learning_rate=0.001):
    """
    Build a CNN model for emotion classification.
    
    Architecture:
        Input
        ↓
        Conv2D (64) → BatchNorm → ReLU → MaxPool → Dropout
        ↓
        Conv2D (128) → BatchNorm → ReLU → MaxPool → Dropout
        ↓
        Conv2D (256) → BatchNorm → ReLU → MaxPool → Dropout
        ↓
        Flatten
        ↓
        Dense (256) → ReLU → Dropout
        ↓
        Dense (128) → ReLU → Dropout
        ↓
        Dense (num_classes) → Softmax
    
    Args:
        input_shape (tuple): Shape of input (n_mfcc, time, 1)
        num_classes (int): Number of emotion classes
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        keras.Model: Compiled CNN model
    """
    model = Sequential(name="CNN_EmotionRecognition")
    
    # Block 1
    model.add(Conv2D(
        64, (3, 3),
        padding='same',
        kernel_regularizer=l2(0.001),
        input_shape=input_shape
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
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
    
    logger.info(f"CNN Model built with input shape {input_shape}")
    model.summary()
    
    return model
