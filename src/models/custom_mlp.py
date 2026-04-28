import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight


def compute_class_weights(y_train):
    """
    Computes balanced class weights from a training label array.
    Returns a dict {class_index: weight} ready to pass to model.fit(class_weight=...).

    Formula: weight_k = total_samples / (n_classes * count_k)
    Minority classes receive higher weights, penalising the loss more
    when the model misclassifies them — effectively correcting for imbalance
    without altering the training data.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))


def build_custom_model(input_columns):
    """
    Constructs a Custom Multi-Layer Perceptron (MLP) for MIMIC-IV clinical data.
    """
    model = Sequential()

    # LAYER 1: The Input Layer
    # Dynamically accepts the exact number of features in the matrix
    model.add(Dense(128, activation='relu', input_shape=(input_columns,)))

    # LAYER 2: First Dropout (Overfitting Protection)
    # Randomly turns off 30% of neurons so it doesn't memorize noisy ICU data
    model.add(Dropout(0.3))

    # LAYER 3: The Hidden Core
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    # LAYER 4: Final Feature Compression
    model.add(Dense(32, activation='relu'))

    # LAYER 5: The Output Layer
    # Sigmoid for binary: outputs probability of the positive class
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['AUC']
    )

    return model


def get_training_callbacks():
    """
    Creates the rules for how the custom model behaves during training.
    Early Stopping: if validation loss doesn't improve for 10 rounds,
    stop and restore the best weights.
    """
    return [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]