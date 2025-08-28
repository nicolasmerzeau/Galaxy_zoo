from galaxy_zoo.logic.model import model_full_pipeline, model_ovr_pipeline
from galaxy_zoo.models.model_tests import model_small_ovr, model_medium_ovr, model_small
from galaxy_zoo.logic.registry import save_model
import pandas as pd
import os
from galaxy_zoo.logic.params import *
import time
import pickle
from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage


target_names = {
    0: "ELLIPTICAL",
    1: "SPIRAL",
    2: "EDGE_CIGAR",
    -1: "ALL",
}
params = {
    'IMG_SIZE': [256],
    'NB_DATA': [10],
    "TEST_SIZE": 0.3,
    "EPOCHS": [10],
}

models = [
    {
        "MODEL_FUNC": model_small,
        "OVR": False,
    },
    {
        "MODEL_FUNC": [model_small_ovr],
        "OVR": True,
        "TARGET_CLASS": [0,1,2],
    }
]

def create_model_name(ovr, img_size, nb_img, epochs, model_func, target = -1):
    if ovr:
        return f"TARGET_{target_names[target]}_{model_func.__name__.upper()}_{img_size}-{img_size}X{nb_img}_EPOCHS_{epochs}"
    else:
        return f"3_CAT_{model_func.__name__.upper()}_{img_size}-{img_size}X{nb_img}_EPOCHS_{epochs}"


def run_models(params=params, models=models):

    metrics = {}

    for img_size in params['IMG_SIZE']:
        for nb_data in params['NB_DATA']:
            input_shape = (img_size, img_size, 3)
            for epochs in params['EPOCHS']:
                for mod in models:
                    if mod['OVR']:
                        for model_func in mod['MODEL_FUNC']:

                            for target in mod['TARGET_CLASS']:
                                model_name = create_model_name(
                                    True,
                                    img_size,
                                    nb_data,
                                    epochs,
                                    model_func,
                                    target
                                )

                                res, model = model_ovr_pipeline(
                                    nb_data,
                                    target,
                                    epochs,
                                    model_func,
                                    input_shape,
                                    metrics_only = True
                                )
                                metrics[model_name] = res
                                save_model(model)
                    else:
                        model_name = create_model_name(
                            False,
                            img_size,
                            nb_data,
                            epochs,
                            mod['MODEL_FUNC'],
                        )
                        res, model = model_full_pipeline(
                            nb_data,
                            epochs,
                            mod['MODEL_FUNC'],
                            input_shape,
                            metrics_only = True
                        )
                        metrics[model_name] = res
                        save_model(model)

    for name, eval in metrics.items():
        print(f"🎯 {name} ———————————————————————————————\n")
        print(eval)
        print('\n—————————————————————————————————————————')

    df_results = pd.DataFrame(metrics).T
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"{timestamp}.csv"

    if MODEL_TARGET == "gcs":
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/metrics/{model_filename}")
        df_results.to_csv(
            blob,
            index=True,
        )
        print("✅ Model saved to GCS")

    else :
        model_path = os.path.join(LOCAL_REGISTRY_PATH, f"{timestamp}.csv")
        df_results.to_csv(model_path, index=True)

        print("✅ Model saved locally")

    print(df_results)

    return metrics


if __name__=="__main__":
    run_models()
