from galaxy_zoo.logic.model import model_full_pipeline, model_ovr_pipeline
from galaxy_zoo.models.model_tests import model_small_ovr, model_medium_ovr, model_small

target_names = {
    0: "ELLIPTICAL",
    1: "SPIRAL",
    2: "EDGE_CIGAR",
}
params = {
    'IMG_SIZE': 256,
    'RANDOM_STATE': 42,
    'NB_DATA': 100,
    "TEST_SIZE": 0.3,
    "EPOCHS": 5,
}

models = [
    {
        "MODEL_NAME": "BASE_LINE_3_CLASSES",
        "MODEL_FUNC": model_small,
        "OVR": False,
        "TARGET_CLASS": 0,
    },
    {
        "MODEL_NAME": "BASE_LINE_SPIRAL",
        "MODEL_FUNC": model_small_ovr,
        "OVR": True,
        "TARGET_CLASS": 0,
    },
    {
        "MODEL_NAME": "BASE_LINE_ELLIPTICAL",
        "MODEL_FUNC": model_small_ovr,
        "OVR": True,
        "TARGET_CLASS": 1,
    },
    {
        "MODEL_NAME": "BASE_LINE_EDGE",
        "MODEL_FUNC": model_small_ovr,
        "OVR": True,
        "TARGET_CLASS": 2,
    },

]
def create_model_name(mod, ovr = True, ):
    if ovr:
        return f"OVR_TARGET_{target_names[mod['TARGET_CLASS']]}_{mod['MODEL_FUNC'].__name__.upper()}"
    else:
        return f"3_CAT_{mod['MODEL_FUNC'].__name__.upper()}"


def run_models(params=params, models=models):

    input_shape = (params['IMG_SIZE'], params['IMG_SIZE'], 3)
    nb_data = params['NB_DATA']
    epochs = params['EPOCHS']

    metrics = {}

    for mod in models:
        model_name = create_model_name(mod, mod['OVR'])
        if mod['OVR']:
            res = model_ovr_pipeline(
                nb_data,
                mod['TARGET_CLASS'],
                epochs,
                mod['MODEL_FUNC'],
                input_shape,
                metrics_only = True
            )
            metrics[model_name] = res
        else:
            res = model_full_pipeline(
                nb_data,
                epochs,
                mod['MODEL_FUNC'],
                input_shape,
                metrics_only = True
            )
            metrics[model_name] = res

    for name, eval in metrics.items():
        print(f"🎯 {name} ———————————————————————————————\n")
        print(eval)
        print('\n—————————————————————————————————————————')

    return metrics
