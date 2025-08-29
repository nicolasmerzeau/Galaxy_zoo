from keras import layers

def model_wrapper(model_func, input_shape, ovr):
    """
    Wraps a model-building function to append a final classification layer.

    Parameters:
        model_func (callable): A function that takes `input_shape` and returns a Keras model.
        input_shape (tuple): The shape of the input data for the model.
        ovr (bool): If True, adds a single-unit sigmoid output layer for binary classification (one-vs-rest).
                    If False, adds a three-unit softmax output layer for multi-class classification.

    Returns:
        keras.Model: The constructed Keras model with the appropriate output layer.
    """

    model = model_func(input_shape)

    if ovr:
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    model.add(layers.Dense(3, activation="softmax"))

    return model
