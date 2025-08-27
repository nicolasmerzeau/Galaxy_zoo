import pandas as pd
import numpy as np
import tensorflow as tf
from keras import models, optimizers
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from typing import Tuple, Dict, Any
from galaxy_zoo.logic.data import load_and_preprocess_data, generate_image_df
from galaxy_zoo.models.Kani.model_02 import create_model

INPUT_SHAPE = (256,256, 3)

# def init_model(input_shape= INPUT_SHAPE, dropout_rate: float = 0.5) -> models.Sequential:
#     """
#     Builds a deep convolutional neural network (CNN) model for binary image classification.

#     This model consists of multiple convolutional blocks with batch normalization, max pooling, and dropout layers to prevent overfitting.
#     It uses global average pooling before fully connected layers, followed by a binary output layer with sigmoid activation.
#     The model is compiled with Adam optimizer, binary cross-entropy loss, and metrics suitable for imbalanced datasets.

#     Args:
#         input_shape (tuple): Shape of the input images, default is (256,256, 3).
#         dropout_rate (float): Dropout rate for regularization in fully connected layers, default is 0.5.

#     Returns:
#         models.Sequential: Compiled Keras Sequential model ready for training.
#     """

#     model = create_model(input_shape, dropout_rate)

#     # Compilation avec des métriques adaptées au déséquilibre de classes
#     model.compile(
#         optimizer=optimizers.Adam(learning_rate=0.001),
#         loss='binary_crossentropy',
#         metrics=[
#             'accuracy',
#             tf.keras.metrics.Precision(name='precision'),
#             tf.keras.metrics.Recall(name='recall'),
#             tf.keras.metrics.AUC(name='auc')
#         ]
#     )

#     return model

def init_model_custom(model_func = create_model, input_shape= INPUT_SHAPE, dropout_rate: float = 0.5) -> models.Sequential:
    """
    Builds a deep convolutional neural network (CNN) model for binary image classification.

    This model consists of multiple convolutional blocks with batch normalization, max pooling, and dropout layers to prevent overfitting.
    It uses global average pooling before fully connected layers, followed by a binary output layer with sigmoid activation.
    The model is compiled with Adam optimizer, binary cross-entropy loss, and metrics suitable for imbalanced datasets.

    Args:
        input_shape (tuple): Shape of the input images, default is (256,256, 3).
        dropout_rate (float): Dropout rate for regularization in fully connected layers, default is 0.5.

    Returns:
        models.Sequential: Compiled Keras Sequential model ready for training.
    """

    model = model_func(input_shape, dropout_rate)

    # Compilation avec des métriques adaptées au déséquilibre de classes
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model

def train_model(df: pd.DataFrame,
                model_func=create_model,
                input_shape=(256, 256, 3),
                target_class=0,
                test_size: float=0.2,
                epochs: int=3,
                batch_size: int=32,
                patience: int=5) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Trains a convolutional neural network (CNN) model using a One-vs-Rest strategy for a specified target class.
        Args:
            df (pd.DataFrame): Input dataframe containing features and labels.
            target_class (int, optional): The class to train the One-vs-Rest classifier for. Defaults to 0.
            test_size (float, optional): Fraction of the data to use for validation. Defaults to 0.2.
            epochs (int, optional): Number of training epochs. Defaults to 3.
            batch_size (int, optional): Size of each training batch. Defaults to 32.
            patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 5.
        Returns:
            Tuple[tf.keras.Model, Dict[str, Any]]: The trained model and a dictionary containing the training history.
    """

    print(f"Entraînement One vs Rest pour la classe {target_class}")

    # Charger et préprocesser les données
    X, y = load_and_preprocess_data(df)

    # Division train/validation stratifiée
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42,
        # X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Construire le modèle
    model = init_model_custom(model_func, input_shape)

    # # Calculer le poids des classes pour gérer le déséquilibre
    # pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    # Callbacks
    es = EarlyStopping(
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    # Entraînement
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[es],
        verbose=1,
        # class_weight={0: 1.0, 1: pos_weight}  # Gérer le déséquilibre
    )

    return model, history.history


def evaluate_model(df: pd.DataFrame, model, target_class = 0, threshold = 0.5) -> Dict[str, float]:
    """
    Evaluates a trained classification model on the provided dataset and computes key metrics.
    Args:
        df (pd.DataFrame): DataFrame containing the input data for evaluation.
        model: Trained Keras model to be evaluated.
        target_class (int, optional): The target class to evaluate against the rest. Defaults to 0.
    Returns:
        Tuple[Dict[str, float], np.ndarray, np.ndarray]:
            - metrics (dict): Dictionary containing loss, accuracy, precision, recall, AUC, and F1-score.
            - y (np.ndarray): True labels.
            - y_pred (np.ndarray): Predicted labels (binary).
    """

    X, y = load_and_preprocess_data(df)

    # Prédictions
    y_pred_prob = model.predict(X, verbose=0)
    y_pred = (y_pred_prob > threshold).astype(int).flatten()

    # Métriques
    results = model.evaluate(X, y, verbose=0)

    metrics = {
        'loss': results[0],
        'accuracy': results[1],
        'precision': results[2],
        'recall': results[3],
        'auc': results[4]
    }

    # F1-score manuel
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    metrics['f1_score'] = f1

    print(f"Métriques d'évaluation (classe {target_class} vs Rest):")
    for metric, value in metrics.items():
        print(f"   {metric.capitalize()}: {value:.4f}")

    return metrics, y, y_pred


def plot_results(history, target_class = "Elliptical"):
    """Affiche les résultats d'entraînement et évaluation"""

    # Courbes d'entraînement
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Résultats d\'entraînement - Classe {target_class} vs Rest', fontsize=16)

    # Loss
    axes[0, 0].plot(history['loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()

    # Accuracy
    axes[0, 1].plot(history['accuracy'], label='Train')
    axes[0, 1].plot(history['val_accuracy'], label='Validation')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()

    # Precision
    axes[1, 0].plot(history['precision'], label='Train')
    axes[1, 0].plot(history['val_precision'], label='Validation')
    axes[1, 0].set_title('Precision')
    axes[1, 0].legend()

    # F1 approximation (Precision * Recall)
    train_f1_approx = np.array(history['precision']) * np.array(history['recall'])
    val_f1_approx = np.array(history['val_precision']) * np.array(history['val_recall'])
    axes[1, 1].plot(train_f1_approx, label='Train F1 (approx)')
    axes[1, 1].plot(val_f1_approx, label='Val F1 (approx)')
    axes[1, 1].set_title('F1 Score (approximation)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, target_class = 0):
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Autres', f'Classe {target_class}'],
                yticklabels=[f'Autres', f'Classe {target_class}'])
    plt.title(f'Matrice de Confusion - Classe {target_class} vs Rest')
    plt.show()

    # Rapport détaillé
    print("\n Rapport de classification:")
    print(classification_report(y_true, y_pred,
                            target_names=[f'Autres', f'Classe {target_class}']))

def model_full_pipeline(
    nb_data = 1000,
    target_class = 0,
    epochs = 10,
    model_func = create_model,
    threshold=0.5
) -> Tuple[pd.DataFrame, tf.keras.Model, Dict[str, Any]]:
    """
    Executes the full pipeline for training and evaluating a CNN model on galaxy images.
    This function generates a DataFrame of image data, trains a model using the specified
    model creation function, plots training results, evaluates the model, and plots the
    confusion matrix.
    Args:
        nb_data (int, optional): Number of data samples to generate. Defaults to 1000.
        target_class (int, optional): Target class label for classification. Defaults to 0.
        model_func (Callable, optional): Function to create the model architecture. Defaults to create_model.
    Returns:
        Tuple[pd.DataFrame, tf.keras.Model, Dict[str, Any]]:
            - df: DataFrame containing image data and labels.
            - model: Trained Keras model.
            - history: Dictionary containing training history and metrics.
    """
    df = generate_image_df(nb_data, target_class)
    model, history = train_model(df, model_func, target_class=target_class, epochs=epochs)
    plot_results(history, target_class)
    metrics, y_true, y_pred = evaluate_model(df, model, target_class, threshold=threshold)
    plot_confusion_matrix(y_true, y_pred, target_class)

    return df, model, history
