from keras import layers, models, Input
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model

def model_vgg(input_shape):

    base_model = VGG16(weights="imagenet", include_top=False)
    base_model.trainable = False

    model = Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Lambda(preprocess_input, name="vgg16_preprocess"))
    model.add(base_model)                          # on empile le modèle VGG16 comme un "layer"
    model.add(layers.Flatten())                    # (ou GlobalAveragePooling2D() ?)
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.3))

    return model
