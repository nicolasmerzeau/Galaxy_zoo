from keras import layers

def model_wrapper(model_func, input_shape, ovr, num_classes):
    """
    Wraps a model creation function, adding a final Dense layer based on the 'ovr' flag.
    Parameters:
        model_func (callable): A function that returns a Keras model given an input shape.
        input_shape (tuple): The shape of the input data, typically (height, width, channels).
        ovr (bool): If True, adds a Dense layer with 1 unit and sigmoid activation for binary classification.
                    If False, adds a Dense layer with units equal to num_classes and softmax activation for multi-class classification.
    Returns:
        keras.Model: The constructed Keras model with the appropriate output layer.
    """

    model = model_func(input_shape)

    if ovr:
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    model.add(layers.Dense(num_classes, activation="softmax"))

    return model
