#Imports pour le modele

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras import Input, layers, optimizers, callbacks
from keras import optimizers
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

import sys, os
sys.path.append(os.path.abspath("../src"))
from galaxy_zoo.logic.data import load_data


# Import de X et y -> Ici 1000 par categorie
df_images = load_data(1000)

# Conversion de X et y
X= df_images["image"]
X = np.stack(X.to_numpy(), axis = 0)
y= df_images["label"]
y_cat= to_categorical(y)


# Initialisation d'un modèle
def initialize_model():

    model = Sequential()

    #Type de donnees d'entree
    model.add(Input((424, 424, 3)))

    model.add(layers.Rescaling(1./255))

    # Architecture test
    model.add(layers.Conv2D(8, (4, 4), activation="relu", padding="same"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    ### Second Convolution & MaxPooling

    model.add(layers.Conv2D(16, (3, 3), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    ### Third Convolution & MaxPooling

    model.add(layers.Conv2D(filters = 64, kernel_size = (3,3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same") )

    model.add(layers.Flatten())

    #Classification en sortie en 3 classes
    model.add(layers.Dense(3, activation="softmax"))

    return model

# Compilation du modèle
def compile_model(model):
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    return model

#Code pour fit et run le modèle

model_small = initialize_model()
model_small = compile_model(model)

es = EarlyStopping(patience = 5, verbose = 2)

history_small = model_small.fit(X, y_cat,
                    validation_split = 0.3,
                    callbacks = [es],
                    epochs = 20,
                    batch_size = 128)

# Avec ces paramètres on obtient une val accuracy de 0.68 (pour 1000 par categorie)
# Avec ces paramètres on obtient une val accuracy de 0.70 (pour 2000 par categorie- batch size 128)
# Avec ces paramètres on obtient une val accuracy de 0.73 (pour 2000 par categorie- batch size 32)




# Fonction pour plot history

def plot_history(history, title='', axs=None, exp_name=""):
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

    ax2.plot(history.history['accuracy'], label='train accuracy'  + exp_name)
    ax2.plot(history.history['val_accuracy'], label='val accuracy'  + exp_name)
    ax2.set_ylim(0.25, 1.)
    ax2.set_title('Accuracy')
    ax2.legend()
    return (ax1, ax2)

# Plot de history

plot_history(history_small)
plt.show()
