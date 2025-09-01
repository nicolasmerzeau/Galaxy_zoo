#Imports pour le modele

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras import Input, layers, optimizers, callbacks
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

from galaxy_zoo.logic.data import generate_image_df, load_and_preprocess_data
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from galaxy_zoo.logic.registry import save_model

# Fonction pour plot history

def plot_history2(history, title='', axs=None, exp_name=""):
    if axs is not None:
        ax1, ax2 = axs
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if len(exp_name) > 0 and exp_name[0] != '_':
        exp_name = '_' + exp_name
    ax1.plot(history.history['loss'], label = 'train' + exp_name)
    ax1.plot(history.history['val_loss'], label = 'val' + exp_name)
    ax1.set_ylim(0., 2.2)
    ax1.set_title('loss')
    ax1.legend()
    ax1.set_xlabel("Epoch")

    ax2.plot(history.history['precision'], label='train precision'  + exp_name)
    ax2.plot(history.history['val_precision'], label='val precision'  + exp_name)
    ax2.set_ylim(0.25, 1.)
    ax2.set_title('precision')
    ax2.legend()
    ax2.set_xlabel("Epoch")
    return (ax1, ax2)

# Plot de history

# plot_history2(history_small)
# plt.show()

if __name__=="__main__":

    a= int(input("Entrez le nombre de galaxies voulues - par classe égales = "))
    b= int(input("Entrez la target size des images -default 424 - x= "))
    df = generate_image_df(nb_data = a) # default values
    X, y = load_and_preprocess_data(df, False, target_size=(b,b))

# Initialisation du modèle

    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(b, b, 3))

    # First step is to initialize the VGG16 model but without the top as we'll adapt it to our problem
    inputs = Input(shape=(b,b, 3))

    x = inputs
    x = preprocess_input(x) # Then a preprocessing layer specifically designed for the VGG16
    x = base_model(x) # Then our transfer learning model

    x = layers.Flatten()(x) # Followed by our custom dense layers, tailored to our binary task

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    pred = layers.Dense(3, activation="softmax")(x)

        # We use the keras Functional API to create our keras model

    model_VGG16 = Model(inputs=inputs , outputs=pred)

        # And we freeze the VGG16 model

    base_model.trainable = False

    adam = optimizers.Adam(learning_rate=0.001)
    model_VGG16.compile(loss='categorical_crossentropy',
                    optimizer=adam,
                    metrics=['Precision'])

    MODEL = "model_VGG16.keras"

    modelCheckpoint = callbacks.ModelCheckpoint(MODEL,
                                                monitor="val_loss",
                                                verbose=0,
                                                save_best_only=True)

    LRreducer = callbacks.ReduceLROnPlateau(monitor="val_loss",
                                            factor=0.1,
                                            patience=3,
                                            verbose=1,
                                            min_lr=0)

    EarlyStopper = callbacks.EarlyStopping(monitor='val_loss',
                                        patience=5,
                                        verbose=0,
                                        restore_best_weights=True)

    history_3 = model_VGG16.fit(
            X,y,
            epochs=15,
            validation_split=0.3,
            callbacks = [modelCheckpoint, LRreducer, EarlyStopper])

    save_model(model_VGG16, model_name="VGG16", history=history_3)


    # Avec ces paramètres on obtient une val accuracy de 0.86 (pour 1000 images par categorie)
