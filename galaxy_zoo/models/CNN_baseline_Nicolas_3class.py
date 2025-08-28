#Imports pour le modele

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras import Input, layers, optimizers, callbacks
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from galaxy_zoo.logic.registry import save_model

from galaxy_zoo.logic.data import generate_image_df, load_and_preprocess_data


# Initialisation d'un modèle
def initialize_model(b):

    model = Sequential()

    #Type de donnees d'entree
    model.add(Input((b,b,3)))

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
                  metrics = ['Precision'])
    return model


# Avec ces paramètres on obtient une val accuracy de 0.68 (pour 1000 par categorie)
# Avec ces paramètres on obtient une val accuracy de 0.70 (pour 2000 par categorie- batch size 128)
# Avec ces paramètres on obtient une val accuracy de 0.73 (pour 2000 par categorie- batch size 32)
# Avec ces paramètres on obtient une val precision de 0.78 (pour 2000 par categorie- batch size 32)



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

    model = initialize_model(b)
    model_small = compile_model(model)

    es = EarlyStopping(patience = 5, verbose = 2,restore_best_weights=True)

    history_small = model_small.fit(X, y,
                    validation_split = 0.3,
                    callbacks = [es],
                    epochs = 20,
                    batch_size = 32)


    model_small.save(f"galaxy/logs/model_tests/model_small_NM_{b}_{a}.keras")
    save_model(model_small)
