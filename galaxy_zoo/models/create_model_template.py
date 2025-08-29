from keras import layers, models, Input
from keras.models import Sequential, Model

def create_model_template(input_shape):

    model = Sequential()
    model.add(Input(input_shape))


    ## -----> Votre modèle ici <------
    ## -----> Votre modèle ici <------
    ## -----> Votre modèle ici <------


    # ——————————————————————————————————————————————————————————————————————————————

    # ⚠️ model.add(layers.Dense(3, activation="softmax")) ----- PAS NECESSAIRE
    # ⚠️ ne pas mettre de couche de sortie , elle est ajoutée automatiquement dans model_wrapper
    # ——————————————————————————————————————————————————————————————————————————————


    return model
