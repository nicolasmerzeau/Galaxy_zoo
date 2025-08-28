from keras import layers, models

# ex modèle 02
def model_small_ovr(input_shape):

    return models.Sequential([
        layers.Input(input_shape),
        layers.Conv2D(32,3,activation="relu"),
        layers.MaxPool2D(),

        layers.Conv2D(64,3,activation="relu"),
        layers.MaxPool2D(),

        layers.Conv2D(16, 3,activation="relu"),

        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),
    ])
