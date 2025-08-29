from galaxy_zoo.logic.params import *
import glob
import os
import time
import pickle
from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage
from typing import Union, Optional
import json



def save_model(model: keras.Model = None, model_name: str = None, history: Union[keras.callbacks.History, dict]=None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")



    # Save model locally
    if model_name :
        model_path = os.path.join(LOCAL_REGISTRY_PATH, "saved_models", f"{timestamp}{model_name}.h5")
        history_path = os.path.join(LOCAL_REGISTRY_PATH, "saved_models", f"{timestamp}{model_name}hist.json")
    else :
        model_path = os.path.join(LOCAL_REGISTRY_PATH, "saved_models", f"{timestamp}.h5")
        history_path = os.path.join(LOCAL_REGISTRY_PATH, "saved_models", f"{timestamp}hist.json")

    model.save(model_path)
    model_filename = model_path.split("/")[-1]

    print("✅ Model saved locally")

    if history is not None:
        # Supporte keras.callbacks.History ou un dict déjà prêt
        if isinstance(history, keras.callbacks.History):
            history_dict = {
                "params": history.params,
                "epoch": history.epoch,
                "history": history.history  # dict métriques -> listes
            }
        elif isinstance(history, dict):
            # On normalise dans une enveloppe
            # Vous pouvez aussi écrire directement le dict si déjà au bon format
            history_dict = {"history": history}
        else:
            raise TypeError(
                "history must be a keras.callbacks.History or a dict of metrics."
            )

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_dict, f, ensure_ascii=False, indent=2)

        history_filename = os.path.basename(history_path)
        print("✅ History saved locally ->", history_path)
    else:
        history_filename = None
        print("ℹ️ No history provided; skipped local history save.")

    if MODEL_TARGET == "gcs":

        # Upload du model

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        # Upload de l'history si disponible
        if history_filename is not None:
            blob_hist = bucket.blob(f"models/{history_filename}")
            blob_hist.upload_from_filename(history_path)
            print("✅ History saved to GCS ->")



        return model_filename

    return model_filename




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




def load_history(history_name: Optional[str] = None, stage: str = "Production") -> Optional[dict]:
    """
    Charger un history (JSON) :
    - Si MODEL_TARGET == "local" : charge un history précis par son nom (ex: "20250827-154230_history.json"),
      sinon le plus récent si history_name=None.
    - Si MODEL_TARGET == "gcs"   : idem mais depuis ton bucket GCS (prefix "models/").
    - Si aucun fichier trouvé : retourne None.
    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad training history from local registry..." + Style.RESET_ALL)
        local_dir = os.path.join(LOCAL_REGISTRY_PATH, "saved_models")

        if not os.path.isdir(local_dir):
            print("❌ Local models directory not found")
            return None

        # 1) Déterminer le chemin du JSON à charger
        if history_name:
            if not history_name.endswith(".json"):
                print(f"❌ Invalid file type: {history_name}. Expected a .json file")
                return None
            history_path = os.path.join(local_dir, history_name)
            if not os.path.exists(history_path):
                print(f"❌ History {history_name} not found locally")
                return None
        else:
            # Prendre le plus récent JSON
            json_candidates = sorted(glob.glob(os.path.join(local_dir, "*_history.json")))
            if not json_candidates:
                print("❌ No local history JSON found")
                return None
            history_path = json_candidates[-1]

        # 2) Charger le JSON
        with open(history_path, "r", encoding="utf-8") as f:
            history_dict = json.load(f)

        print(f"✅ History loaded from {history_path}")
        return history_dict

    elif MODEL_TARGET == "gcs":
        print(Fore.BLUE + f"\nLoad training history from GCS..." + Style.RESET_ALL)
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        if history_name:
            if not history_name.endswith(".json"):
                print(f"❌ Invalid file type: {history_name}. Expected a .json file")
                return None
            hist_blob_name = f"models/{history_name}"
            hist_blob = bucket.blob(hist_blob_name)
            if not hist_blob.exists(client):
                print(f"❌ History not found in GCS: gs://{BUCKET_NAME}/{hist_blob_name}")
                return None
        else:
            # Prendre le plus récent *_history.json
            blobs = list(bucket.list_blobs(prefix="models/"))
            json_blobs = [b for b in blobs if b.name.endswith(".json")]
            if not json_blobs:
                print(f"❌ No history JSON found in GCS bucket {BUCKET_NAME}")
                return None
            hist_blob = max(json_blobs, key=lambda x: x.updated)

        # 2) Télécharger et charger
        local_dir = os.path.join(LOCAL_REGISTRY_PATH, "loaded_models")
        os.makedirs(local_dir, exist_ok=True)
        local_history_path = os.path.join(local_dir, os.path.basename(hist_blob.name))
        hist_blob.download_to_filename(local_history_path)

        with open(local_history_path, "r", encoding="utf-8") as f:
            history_dict = json.load(f)

        print(f"✅ History loaded from GCS: gs://{BUCKET_NAME}/{hist_blob.name}")
        return history_dict

    print("❌ Unknown MODEL_TARGET. Expected 'local' or 'gcs'.")
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
