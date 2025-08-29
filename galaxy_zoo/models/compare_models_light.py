from galaxy_zoo.logic.model import generate_image_df
from galaxy_zoo.logic.model import model_full_pipeline, model_ovr_pipeline
from galaxy_zoo.models.model_tests import model_large_kani, model_small_nicolas
from galaxy_zoo.logic.registry import save_model
import pandas as pd
import os
from galaxy_zoo.logic.params import *
import time
import pickle
from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage
from io import StringIO


target_names = {
    0: "ELLIPTICAL",
    1: "SPIRAL",
    2: "EDGE_CIGAR",
    -1: "ALL",
}
params = {
    'IMG_SIZE': [256],
    'NB_DATA': [50],
    "TEST_SIZE": 0.3,
    "EPOCHS": [10],
}

models = [model_small_nicolas, model_large_kani]
cats = [0, 1, 2]

def create_model_name(ovr, img_size, nb_img, epochs, model_func, target = -1):
    if ovr:
        return f"TARGET_{target_names[target]}_{model_func.__name__.upper()}_{img_size}-{img_size}X{nb_img}_EPOCHS_{epochs}"
    else:
        return f"3_CAT_{model_func.__name__.upper()}_{img_size}-{img_size}X{nb_img}_EPOCHS_{epochs}"


def run_models(params=params, models=models):

    metrics = {}
    for img_size in params['IMG_SIZE']:
        input_shape = (img_size, img_size, 3)

        for nb_data in params['NB_DATA']:
            df = generate_image_df(nb_data)

            for epochs in params['EPOCHS']:
                for model_func in models:
                    # on test chaque model sur : 3 classes + 1 vs Rest * 3
                    # 3 classes
                    model_name = create_model_name(
                        False,
                        img_size,
                        nb_data,
                        epochs,
                        model_func,
                    )
                    res, model = model_full_pipeline(
                        df,
                        epochs,
                        model_func,
                        input_shape,
                        metrics_only = True
                    )
                    h5_name = save_model(model, model_name)
                    metrics[h5_name] = res

                    # OVR
                    for target in cats: # 0, 1, ou 2
                        model_name = create_model_name(
                            True,
                            img_size,
                            nb_data,
                            epochs,
                            model_func,
                            target
                        )

                        res, model = model_ovr_pipeline(
                            df,
                            target,
                            epochs,
                            model_func,
                            input_shape,
                            metrics_only = True
                        )
                        h5_name = save_model(model, model_name)
                        metrics[h5_name] = res

    for name, eval in metrics.items():
        print(f"🎯 {name} ———————————————————————————————\n")
        print(eval)
        print('\n—————————————————————————————————————————')

    df_results = pd.DataFrame(metrics).T
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"{timestamp}.csv"

    if MODEL_TARGET == "gcs":
        csv_buf = StringIO()
        df_results.to_csv(csv_buf, index=True)  # index=True pour garder le nom du modèle
        csv_str = csv_buf.getvalue()

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/metrics/{model_filename}")
        blob.upload_from_string(csv_str, content_type="text/csv")

        print("✅ Model saved to GCS")

    else :
        model_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", f"{timestamp}.csv")
        df_results.to_csv(model_path, index=True)

        print("✅ Model saved locally")

    print(df_results)

    return metrics


if __name__=="__main__":
    run_models()
