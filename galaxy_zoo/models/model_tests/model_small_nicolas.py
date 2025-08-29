from keras import layers, models, Input
from keras.models import Sequential, Model

# ex - modèle nicolas
def model_small_nicolas(input_shape):

    model = Sequential()

    model.add(Input(input_shape))

    model.add(layers.Conv2D(8, (4, 4), activation="relu", padding="same"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters = 64, kernel_size = (3,3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    # model.add(layers.Dense(3, activation="softmax")) ----->  ajouté dans model_wrapper
    return model
