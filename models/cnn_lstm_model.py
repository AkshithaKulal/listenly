"""
cnn_lstm_model.py - Hybrid CNN+LSTM model (best performance).
CNN extracts spatial features, LSTM captures temporal patterns.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    LSTM, Dense, Dropout, Reshape, Bidirectional
)
from tensorflow.keras.regularizers import l2
import logging

logger = logging.getLogger(__name__)


def build_cnn_lstm_model(input_shape, num_classes, learning_rate=0.0005):
    """
    Build a hybrid CNN+LSTM model for emotion classification.

    Input shape: (n_mfcc, time, channels) — channels may be 1 or 3 (MFCC+deltas).
    """
    model = Sequential(name="CNN_LSTM_EmotionRecognition")

    model.add(Conv2D(
        32, (3, 3),
        padding='same',
        activation='relu',
        kernel_regularizer=l2(1e-4),
        input_shape=input_shape,
    ))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # (batch, h, w, c) -> (batch, w, h*c) keep width as time
    shape_before_lstm = model.output_shape[1:]
    new_shape = (shape_before_lstm[1], shape_before_lstm[0] * shape_before_lstm[2])
    model.add(Reshape(new_shape))

    model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4))))
    model.add(Dropout(0.35))
    model.add(Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(1e-4))))
    model.add(Dropout(0.35))

    model.add(Dense(128, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    logger.info(f"CNN+LSTM Model built with input shape {input_shape}")
    model.summary()
    return model
