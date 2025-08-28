from params import *
import glob
import os
import time
import pickle
from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage


def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "saved_models", f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":


        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    return None



def load_model(model_name: str = None, stage="Production") -> keras.Model:
    """
    Charger un modèle Keras :
    - Si MODEL_TARGET == "local" : charge un modèle précis par son nom (ex: "20250827-154230.h5"),
      sinon le plus récent si model_name=None.
    - Si MODEL_TARGET == "gcs"   : idem mais depuis ton bucket GCS.
    - Si aucun modèle trouvé : retourne None.
    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad model from local registry..." + Style.RESET_ALL)

        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "saved_models")

        if model_name:  #  cas où tu donnes le nom du fichier
            model_path = os.path.join(local_model_directory, model_name)
            if not os.path.exists(model_path):
                print(f"❌ Model {model_name} not found locally")
                return None
        else:  # 👈 sinon on prend le plus récent
            local_model_paths = glob.glob(f"{local_model_directory}/*")
            if not local_model_paths:
                return None
            model_path = sorted(local_model_paths)[-1]

        latest_model = keras.models.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")
        return latest_model

    elif MODEL_TARGET == "gcs":
        print(Fore.BLUE + f"\nLoad model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models"))

        if not blobs:
            print(f"❌ No model found in GCS bucket {BUCKET_NAME}")
            return None

        if model_name:  # 👈 tu passes un nom précis
            matches = [b for b in blobs if b.name.endswith(model_name)]
            if not matches:
                print(f"❌ Model {model_name} not found in GCS bucket {BUCKET_NAME}")
                return None
            latest_blob = matches[0]
        else:  # 👈 sinon on prend le plus récent
            latest_blob = max(blobs, key=lambda x: x.updated)
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "loaded_models")
        latest_model_path_to_save = os.path.join(local_model_directory, latest_blob.name.split("/")[-1])
        latest_blob.download_to_filename(latest_model_path_to_save)

        latest_model = keras.models.load_model(latest_model_path_to_save)
        print(f"✅ Model loaded from GCS: {latest_blob.name}")
        return latest_model

    return None


def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")
