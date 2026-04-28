import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

def compute_class_weights(y_train):
    """
    Computes balanced class weights from a training label array.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))

def build_custom_mlp(input_dim, task_type, n_classes, units_1=128, units_2=64, units_3=32, dropout_1=0.3, dropout_2=0.2, lr=0.001):
    """
    Constructs the Custom Multi-Layer Perceptron (MLP) for MIMIC-IV clinical data.
    Dynamically routes architecture for Binary vs Multiclass classification.
    """
    model = Sequential([
        Dense(units_1, activation='relu', input_shape=(input_dim,)),
        Dropout(dropout_1),
        Dense(units_2, activation='relu'),
        Dropout(dropout_2),
        Dense(units_3, activation='relu')
    ])

    if task_type == 'binary':
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['AUC'])
    else:
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model