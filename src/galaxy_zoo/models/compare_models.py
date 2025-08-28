from galaxy_zoo.logic.model import model_full_pipeline, model_ovr_pipeline
from galaxy_zoo.models.model_tests import model_small_ovr, model_medium_ovr, model_small

target_names = {
    0: "ELLIPTICAL",
    1: "SPIRAL",
    2: "EDGE_CIGAR",
    -1: "ALL",
}
params = {
    'IMG_SIZE': [256, 424],
    'NB_DATA': [100],
    "TEST_SIZE": 0.3,
    "EPOCHS": [3],
}

models = [
    {
        "MODEL_FUNC": model_small,
        "OVR": False,
    },
    {
        "MODEL_FUNC": model_small_ovr,
        "OVR": True,
        "TARGET_CLASS": [0,1,2],
    },

]
def create_model_name(mod, img_size, nb_img, epochs, target = -1):
    if mod['OVR']:
        return f"TARGET_{target_names[target]}_{mod['MODEL_FUNC'].__name__.upper()}_{img_size}_{nb_img}_EPOCHS_{epochs}"
    else:
        return f"3_CAT_{mod['MODEL_FUNC'].__name__.upper()}_{img_size}_{nb_img}_EPOCHS_{epochs}"


def run_models(params=params, models=models):

    metrics = {}

    for img_size in params['IMG_SIZE']:
        for nb_data in params['NB_DATA']:
            input_shape = (img_size, img_size, 3)
            for epochs in params['EPOCHS']:
                for mod in models:
                    if mod['OVR']:
                        for target in mod['TARGET_CLASS']:
                            model_name = create_model_name(mod, img_size, nb_data, epochs, target)
                            res = model_ovr_pipeline(
                                nb_data,
                                target,
                                epochs,
                                mod['MODEL_FUNC'],
                                input_shape,
                                metrics_only = True
                            )
                            metrics[model_name] = res
                    else:
                        model_name = create_model_name(mod, img_size, nb_data, epochs)
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
