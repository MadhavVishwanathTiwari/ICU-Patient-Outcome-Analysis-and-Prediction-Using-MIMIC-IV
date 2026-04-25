import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def build_custom_model(input_columns):
    """
    Constructs a Custom Multi-Layer Perceptron (MLP) for MIMIC-IV clinical data.
    """
    # Initialize the neural network
    model = Sequential()
    
    # LAYER 1: The Input Layer
    # It dynamically accepts the exact number of features in your matrix (e.g., 68)
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
    # Uses a sigmoid function to spit out a clean 0% to 100% probability of mortality
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model's learning logic
    model.compile(
        optimizer='adam',                  # The algorithm that updates the weights
        loss='binary_crossentropy',        # The math used to calculate prediction errors
        metrics=['AUC']                    # We track AUC to compare it directly with XGBoost
    )

    return model

def get_training_callbacks():
    """
    Creates the rules for how the custom model behaves during training.
    """
    # Early Stopping: If the validation loss doesn't improve for 10 rounds, 
    # stop training and restore the best version of the model.
    return [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]