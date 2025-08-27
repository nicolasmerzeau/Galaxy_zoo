from keras import layers, models

def create_model_template(input_shape, dropout_rate = 0.25):
    """
    Creates a sequential Keras model template for image classification tasks.
    The model consists of:
    - An initial Conv2D layer with ReLU activation and 'same' padding.
    - A GlobalAveragePooling2D layer to reduce the number of parameters.
    - A Dense layer with ReLU activation and dropout for regularization.
    - An output Dense layer with sigmoid activation for binary classification.
    Parameters:
        input_shape (tuple): Shape of the input images, e.g., (height, width, channels).
        dropout_rate (float): Dropout rate for regularization.
    Returns:
        keras.models.Sequential: The constructed Keras sequential model.
    """

    return models.Sequential([
        # ------------------ Input Layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        # layers.BatchNormalization(),
        # layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        # layers.MaxPooling2D((2, 2)),
        # layers.Dropout(0.25),

        # layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        # layers.BatchNormalization(),
        # layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        # layers.MaxPooling2D((2, 2)),
        # layers.Dropout(0.25),

        # layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        # layers.BatchNormalization(),
        # layers.Dropout(0.25),

        # -------------------- Global Average Pooling au lieu de Flatten -> réduit le nb de paramètres
        layers.GlobalAveragePooling2D(),

        # --------------------- Couches fully connected
        # layers.Dense(512, activation='relu'),
        # layers.BatchNormalization(),
        # layers.Dropout(dropout_rate),

        # layers.Dense(256, activation='relu'),
        # layers.BatchNormalization(),
        # layers.Dropout(dropout_rate),

        layers.Dense(input_shape[1], activation='relu'),
        layers.Dropout(dropout_rate),

        # ---------------------- Couche de sortie binaire
        layers.Dense(1, activation='sigmoid')
    ])
