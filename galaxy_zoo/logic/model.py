import pandas as pd
import numpy as np
import tensorflow as tf
from keras import models, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from typing import Tuple, Dict, Any
from galaxy_zoo.logic.data import load_and_preprocess_data, generate_image_df
from galaxy_zoo.models.model_tests import model_small_nicolas
from galaxy_zoo.models.model_wrapper import model_wrapper
from galaxy_zoo.logic.params import *
from keras.utils import to_categorical

target_names = {
    0: "Elliptical",
    1: "Spiral",
    2: "Edge-on / Cigar",
    -1: "Other"
}

def init_model_custom(model_func = model_small_nicolas, input_shape = INPUT_SHAPE, ovr = True) -> models.Sequential:
    """
    Builds a deep convolutional neural network (CNN) model for binary image classification.

    This model consists of multiple convolutional blocks with batch normalization, max pooling, and dropout layers to prevent overfitting.
    It uses global average pooling before fully connected layers, followed by a binary output layer with sigmoid activation.
    The model is compiled with Adam optimizer, binary cross-entropy loss, and metrics suitable for imbalanced datasets.

    Args:
        input_shape (tuple): Shape of the input images, default is (256,256, 3)
    Returns:
        models.Sequential: Compiled Keras Sequential model ready for training.
    """

    model = model_wrapper(model_func, input_shape, ovr)

    loss = 'categorical_crossentropy'
    if ovr:
        loss = 'binary_crossentropy'

    # Compilation avec des métriques adaptées au déséquilibre de classes
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model

def get_class_weight(y_train):
    """
        Compute class weights for a binary classification problem (One-vs-Rest).

        Parameters
        ----------
        y_train : array-like
            Array of training labels containing only 0 and 1.

        Returns
        -------
        dict
            Dictionary mapping each class (0 and 1) to its computed weight.
    """
    classes = np.unique(y_train)  # [0,1]
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=list(y_train)
    )
    return dict(zip(classes, weights))

def train_model(df: pd.DataFrame,
                model_func=model_small_nicolas,
                input_shape=INPUT_SHAPE,
                target_class=0,
                ovr = True,
                test_size: float=0.2,
                epochs: int=5,
                batch_size: int=32,
                patience: int=5) -> Tuple[tf.keras.Model, Dict[str, Any], np.ndarray, np.ndarray]:

    """
        Trains a Keras model on the provided DataFrame using specified parameters.
        Args:
            df (pd.DataFrame): Input DataFrame containing image data and labels.
            model_func (Callable, optional): Function to create the Keras model. Defaults to model_small_nicolas.
            input_shape (Tuple[int, int, int], optional): Shape of the input images. Defaults to INPUT_SHAPE.
            target_class (int, optional): Target class for One-vs-Rest training. Defaults to 0.
            ovr (bool, optional): If True, performs One-vs-Rest training for the target class. If False, trains on all classes. Defaults to True.
            test_size (float, optional): Fraction of data to use for validation. Defaults to 0.2.
            epochs (int, optional): Number of training epochs. Defaults to 5.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 5.
        Returns:
            Tuple[tf.keras.Model, Dict[str, Any], np.ndarray, np.ndarray]:
                - Trained Keras model.
                - Training history as a dictionary.
                - Preprocessed input data (X).
                - Corresponding labels (y).
    """

    if ovr:
        print(f"Entraînement One vs Rest pour la classe {target_class}")
    else:
        print(f"Entraînement sur les 3 classes")


    # Charger et préprocesser les données
    X, y = load_and_preprocess_data(df, ovr, target_class, target_size=input_shape[:2])

    # Division train/validation stratifiée
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    # Construire le modèle
    model = init_model_custom(model_func, input_shape, ovr=ovr)


    # Calculer le poids des classes pour gérer le déséquilibre
    class_weight = {0:1, 1:1, 2: 1}
    if ovr:
        class_weight = get_class_weight(y_train)

    # Callbacks
    es = EarlyStopping(
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    path = os.path.join(LOCAL_REGISTRY_PATH, "saved_epochs")

    modelCheckpoint = ModelCheckpoint(
        path,
        monitor="val_precision",
        verbose=0,
        save_best_only=True,
        # save_freq=,
    )

    # Entraînement
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[es, modelCheckpoint],
        verbose=1,
        class_weight=class_weight  # Gérer le déséquilibre
    )

    return model, history.history, X, y

def train_model_with_processed_data(
                split_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], # (X_train, X_val, y_train, y_val)
                model_func=model_small_nicolas,
                input_shape=INPUT_SHAPE,
                target_class=0,
                ovr = True,
                epochs: int=5,
                batch_size: int=32,
                patience: int=10) -> Tuple[tf.keras.Model, Dict[str, Any]]:


    if ovr:
        print(f"Entraînement One vs Rest pour la classe {target_class}")
    else:
        print(f"Entraînement sur les 3 classes")


    # Construire le modèle
    model = init_model_custom(model_func, input_shape, ovr=ovr)

    X_train, X_val, y_train, y_val = split_data

    # Calculer le poids des classes pour gérer le déséquilibre
    class_weight = {0:1, 1:1, 2: 1}
    if ovr:
        class_weight = get_class_weight(y_train)

    # Callbacks
    es = EarlyStopping(
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    path = os.path.join(LOCAL_REGISTRY_PATH, "saved_epochs")

    modelCheckpoint = ModelCheckpoint(
        path,
        monitor="val_precision",
        verbose=0,
        save_best_only=True,
        # save_freq=,
    )
    # Entraînement
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[es, modelCheckpoint],
        verbose=1,
        class_weight=class_weight  # Gérer le déséquilibre
        ### save toutes les X epochs
    )

    return model, history.history


def evaluate_model(X, y, model, target_class = 0, threshold = 0.5) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
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

    if target_class == -1 :
        print(f"Métriques d'évaluation:")
    else :
        print(f"Métriques d'évaluation (classe {target_class} vs Rest):")

    for metric, value in metrics.items():
        print(f"   {metric.capitalize()}: {value:.4f}")

    return metrics, y, y_pred, y_pred_prob


def plot_results(history, target_class = 0):
    """Affiche les résultats d'entraînement et évaluation"""

    # Courbes d'entraînement
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    if target_class != -1:
        fig.suptitle(f'Résultats d\'entraînement - Classe {target_names[target_class]} vs Rest', fontsize=16)
    else :
        fig.suptitle(f'Résultats d\'entraînement', fontsize=16)

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

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_ovr(y_true, y_pred, target_class = 0):
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

TARGET_NAMES = ["Elliptical", "Spiral", "Edge-on / Cigar"]

def plot_confusion_matrix(y_true, y_pred):
    print("Shape des prédictions:", y_pred.shape)
    print("Shape des vraies étiquettes:", y_true.shape)
    print("Nombre d'échantillons prédictions:", len(y_pred))
    print("Nombre d'échantillons vraies étiquettes:", len(y_true))
    # Matrice de confusion
    # y_pred = np.argmax(y_pred, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)

    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de Confusion')
    plt.show()

    # Rapport détaillé
    print("\n Rapport de classification:")
    print(classification_report(y_true_labels, y_pred_labels))

def model_ovr_pipeline(
    df: pd.DataFrame,
    target_class = 0,
    epochs = 5,
    model_func = model_small_nicolas,
    input_shape=INPUT_SHAPE,
    threshold = 0.5,
    metrics_only = False
) -> Tuple[pd.DataFrame, tf.keras.Model, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
        Builds and trains a one-vs-rest classification model pipeline for galaxy images.
        This function generates a dataset, trains a model, plots training results,
        evaluates the model, and plots the confusion matrix for the specified target class.
        Args:
            nb_data (int, optional): Number of data samples to generate. Defaults to 1000.
            target_class (int, optional): The target class for one-vs-rest classification. Defaults to 5.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            model_func (Callable, optional): Function to create the model architecture. Defaults to model_small_nicolas.
            input_shape (tuple, optional): Shape of the input images. Defaults to INPUT_SHAPE.
        Returns:
            Tuple[
                pd.DataFrame,         # DataFrame containing generated image data and labels
                tf.keras.Model,       # Trained Keras model
                Dict[str, Any],       # Training history
                np.ndarray,           # Input data used for evaluation
                np.ndarray,           # True labels for evaluation
                np.ndarray            # Predicted labels for evaluation
            ]
    """
    # df = generate_image_df(nb_data, target_class)
    model, history, X, y = train_model(
        df,
        model_func,
        input_shape=input_shape,
        target_class=target_class,
        epochs=epochs
    )

    metrics, y_true, y_pred = evaluate_model(X, y, model, target_class)
    if metrics_only:
        return metrics, model

    plot_results(history, target_class)
    plot_confusion_matrix_ovr(y_true, y_pred, target_class)

    return df, model, history, X, y_true, y_pred


def model_full_pipeline(
    df: pd.DataFrame,
    epochs = 5,
    model_func = model_small_nicolas,
    input_shape=INPUT_SHAPE,
    metrics_only = False
) -> Tuple[pd.DataFrame, tf.keras.Model, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the full pipeline for training and evaluating a machine learning model on generated image data.
    Args:
        nb_data (int, optional): Number of data samples to generate. Defaults to 1000.
        epochs (int, optional): Number of training epochs. Defaults to 5.
        model_func (Callable, optional): Function to create the model architecture. Defaults to `model_small_nicolas`.
        input_shape (tuple, optional): Shape of the input data. Defaults to `INPUT_SHAPE`.
    Returns:
        Tuple[
            pd.DataFrame,         # DataFrame containing generated image data and labels
            tf.keras.Model,       # Trained Keras model
            Dict[str, Any],       # Training history dictionary
            np.ndarray,           # Input data array (X)
            np.ndarray,           # True labels array (y)
            np.ndarray            # Predicted labels array (y_pred)
        ]
    """

    model, history, X, y = train_model(
        df,
        model_func,
        input_shape=input_shape,
        ovr=False,
        epochs=epochs
    )

    metrics, y, y_pred, y_pred_proba = evaluate_model(X, y, model, -1)
    if metrics_only:
        return metrics, model

    plot_results(history, -1)
    plot_confusion_matrix(y, y_pred_proba)

    return df, model, history, X, y, y_pred

def model_ovr_pipeline_from_preproc(
    split_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], # (X_train, X_val, y_train, y_val)
    X,
    y,
    target_class = 0,
    epochs = 5,
    model_func = model_small_nicolas,
    input_shape=INPUT_SHAPE,
    metrics_only = False
) -> Tuple[pd.DataFrame, tf.keras.Model, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    Trains and evaluates a one-vs-rest (OVR) classification model using preprocessed data.
    Args:
        split_data (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
            A tuple containing (X_train, X_val, y_train, y_val) arrays for training and validation.
        X (np.ndarray):
            Input features for evaluation.
        y (np.ndarray):
            True labels for evaluation.
        target_class (int, optional):
            The target class for OVR classification. Defaults to 0.
        epochs (int, optional):
            Number of training epochs. Defaults to 5.
        model_func (Callable, optional):
            Function to create the model architecture. Defaults to model_small_nicolas.
        input_shape (Any, optional):
            Shape of the input data. Defaults to INPUT_SHAPE.
        metrics_only (bool, optional):
            If True, returns only metrics, model, and history. Defaults to False.
    Returns:
        Tuple:
            If metrics_only is True:
                metrics (pd.DataFrame): Evaluation metrics.
                model (tf.keras.Model): Trained Keras model.
                history (Dict[str, Any]): Training history.
            If metrics_only is False:
                model (tf.keras.Model): Trained Keras model.
                history (Dict[str, Any]): Training history.
                X (np.ndarray): Input features.
                y_true (np.ndarray): True labels.
                y_pred (np.ndarray): Predicted labels.
    """
    # df = generate_image_df(nb_data, target_class)
    model, history = train_model_with_processed_data(
        split_data,
        model_func,
        input_shape,
        target_class,
        ovr=True,
        epochs=epochs
    )

    metrics, y_true, y_pred = evaluate_model(X, y, model, target_class)
    if metrics_only:
        return metrics, model, history

    plot_results(history, target_class)
    plot_confusion_matrix(y_true, y_pred, target_class)

    return model, history, X, y_true, y_pred


def model_full_pipeline_from_preproc(
    split_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], # (X_train, X_val, y_train, y_val)
    X,
    y,
    epochs = 5,
    model_func = model_small_nicolas,
    input_shape=INPUT_SHAPE,
    metrics_only = False
) -> Tuple[tf.keras.Model, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:

    """
    Runs the full model pipeline starting from preprocessed data, including training, evaluation, and optional plotting.
    Args:
        split_data (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): Tuple containing (X_train, X_val, y_train, y_val) arrays.
        X: Input features for evaluation.
        y: Target labels for evaluation.
        epochs (int, optional): Number of training epochs. Defaults to 5.
        model_func (callable, optional): Function to create the model architecture. Defaults to model_small_nicolas.
        input_shape: Shape of the input data. Defaults to INPUT_SHAPE.
        metrics_only (bool, optional): If True, only returns metrics, model, and history. Defaults to False.
    Returns:
        Tuple[pd.DataFrame, tf.keras.Model, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
            If metrics_only is True:
                metrics (pd.DataFrame): Evaluation metrics.
                model (tf.keras.Model): Trained Keras model.
                history (Dict[str, Any]): Training history.
            If metrics_only is False:
                model (tf.keras.Model): Trained Keras model.
                history (Dict[str, Any]): Training history.
                X (np.ndarray): Input features.
                y (np.ndarray): True labels.
                y_pred (np.ndarray): Model predictions.
    """

    model, history = train_model_with_processed_data(
        split_data,
        model_func,
        input_shape,
        ovr=False,
        epochs=epochs
    )

    metrics, y, y_pred = evaluate_model(X, y, model, -1)
    if metrics_only:
        return metrics, model, history

    plot_results(history, -1)
    return model, history, X, y, y_pred
