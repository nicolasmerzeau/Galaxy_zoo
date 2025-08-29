from keras import layers, models, Input
from keras.models import Sequential, Model

def model_extralarge_ghalya(input_shape):

    model = Sequential()
    model.add(Input(input_shape))

    model.add(layers.Conv2D(16, (7, 7), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(16, (4, 4), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, (7, 7), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, (4, 4), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (4,4), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (7,7), activation='relu', padding='same'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (4,4), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (4, 4), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, (7, 7), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, (4, 4), activation='relu', padding='same'))

    model.add(layers.Conv2D(16, (7, 7), activation='relu', padding='same'))
    model.add(layers.Conv2D(16, (4, 4), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))


    model.add(layers.Flatten())


    return model
