from keras import layers, models

def create_model(input_shape, dropout_rate = 0.3):
    return models.Sequential([
        layers.Input(input_shape),
        layers.Conv2D(32,3,activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(64,3,activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(128,3,activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation="sigmoid"),
    ])
