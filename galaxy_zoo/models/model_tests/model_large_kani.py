from keras import layers, models, Input
from keras.models import Sequential, Model

def model_large_kani(input_shape):

    model = Sequential()
    model.add(Input(input_shape))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(32, (7, 7), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())

    # model.add(layers.Dense(1, activation='sigmoid')) -----> Ajouté dans model_wrapper

    return model
