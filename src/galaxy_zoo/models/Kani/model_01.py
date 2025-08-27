from keras import layers, models

def create_model(input_shape, dropout_rate):
    return models.Sequential([
        # Premier bloc convolutionnel
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Deuxième bloc convolutionnel
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Global Average Pooling au lieu de Flatten -> réfduit le nb de paramètres
        layers.GlobalAveragePooling2D(),

        # Couches fully connected
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate/2),

        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate/2),

        # Couche de sortie binaire
        layers.Dense(1, activation='sigmoid')
    ])
