import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import os
from PIL import Image
import tensorflow as tf
from keras.utils import to_categorical
from google.cloud import storage
from galaxy_zoo.logic.params import *

FILE_PATH = os.path.join(ROOT_DATA, "gz2_train_catalog.parquet")

# target_names = {
#     0: "Elliptical",
#     1: "Spiral",
#     2: "Edge-on / Cigar",
#     -1: "Other"
# }

def generate_image_df(nb_data = 2000) :
    """
    Generates a balanced DataFrame of image file paths and their corresponding labels for use in experiments.
    This function loads experimental data from a parquet file, maps labels to simplified targets,
    filters out invalid targets, and balances the dataset by sampling an equal number of images per class.
    It then constructs the full file paths for each image based on their filename prefixes.
    Args:
        nb_data (int, optional): Number of samples to select per class. Defaults to 2000.
    Returns:
        pandas.DataFrame: A DataFrame containing the following columns:
            - 'prefix': The first 6 characters of the filename, used to locate the subfolder.
            - 'filename': The image filename.
            - 'path': The full path to the image file.
            - 'simple_target': The simplified target label for the image.
    """

    # Load data updated returns img path instead of matrices


    # Lecture
    df_experiment = pd.read_parquet(FILE_PATH)

    df_experiment["simple_target"] = df_experiment["label"].map(LABEL_MAP)
    df_experiment = df_experiment[df_experiment["simple_target"] != -1]
    df_balanced = (
        df_experiment.groupby("simple_target")
        .apply(lambda x: x.sample(n=nb_data, random_state=RANDOM_STATE))
        .reset_index(drop=True)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    # Dossier à parcourir
    folder_path = os.path.join(ROOT_DATA, "images")

    # Recherche du sous dossier à partir des 6 premiers caractères du filename
    df_balanced['prefix'] = df_balanced['filename'].str.slice(0, 6)
    df_balanced['path'] = df_balanced.apply(lambda r: os.path.join(folder_path, r['prefix'], r['filename']), axis=1)

    return df_balanced[['prefix', 'filename', 'path', 'simple_target' ]]


def load_and_preprocess_data(df: pd.DataFrame,
                            # nb_data: int = 2000,
                            ovr: bool = True,
                            target_class: int = 0,
                            target_size: Tuple[int, int] = (IMG_SIZE, IMG_SIZE),
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
        Loads and preprocesses images from a DataFrame, resizing them and creating binary labels for a specified target class.
        Args:
            df (pd.DataFrame): DataFrame containing image file paths in the 'path' column and class labels in the 'simple_target' column.
            target_class (int, optional): The class to be considered as positive (1) in the binary classification. Defaults to 0.
            target_size (Tuple[int, int], optional): Desired size (height, width) to resize images to. Defaults to (256, 256).
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - X: Array of preprocessed images of shape (num_images, height, width, channels).
                - y: Array of binary labels (1 for target_class, 0 for others).
    """
    images = []
    labels = []

    # print_debug("load_and_preprocess_data", f"input_shape --> {target_size}")
    # df = generate_image_df(nb_data)
    # print(f"Chargement de {len(df)} images...")

    for idx, row in df.iterrows():
        failed_loads = 0
        try:

            # Charger l'image
            image_path = row['path']
            if not os.path.exists(image_path):
                print(f"⚠️  Fichier introuvable: {image_path}")
                failed_loads += 1
                continue

            img = tf.io.read_file(image_path)
            img = tf.io.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, target_size)
            img = tf.image.convert_image_dtype(img, tf.float32)

            images.append(img)
            original_label = int(row['simple_target'])
            num_classes = 3
            if ovr:
                # Créer le label binary One vs Rest
                binary_label = 1 if original_label == target_class else 0
                num_classes = 2
                labels.append(binary_label)
            else:
                labels.append(original_label)

        except Exception as e:
                print(f"❌ Erreur lors du chargement de {row['path']}: {str(e)}")
                failed_loads += 1
                continue

        if failed_loads > 0:
            print(f"⚠️  {failed_loads} images n'ont pas pu être chargées")

    X = np.array(images)
    y = to_categorical(labels, num_classes=num_classes)
    if ovr:
        y = np.array(labels)

    print(f"{len(X)} images chargées avec succès")
    print(f"   Shape des images: {X.shape}")

    return X, y


def load_data(nbr_data = 2000) :


    current_dir = os.path.dirname(__file__) #Garde le chemin vers data.py peut import où la fonction est appelé

    root_data = os.path.abspath(os.path.join(current_dir, "../../../raw_data")) # on revient a src puis galaxy_zoo




    # Chemin vers ton fichier
    file_path = os.path.join(root_data, "gz2_train_catalog.parquet")

# Lecture
    df_experiment = pd.read_parquet(file_path)
    dict_mapping = {
    0:0,
    1:0,
    2:2,
    3:2,
    4:1,
    5:1,
    6:-1,
    -1: -1
    }
    target_names = {
    0: "Elliptical",
    1: "Spiral",
    2: "Edge-on / Cigar",
    -1: "Other"
    }
    df_experiment["simple_target"] = df_experiment["label"].map(dict_mapping)
    df_experiment = df_experiment[df_experiment["simple_target"]!=-1] # on enleve dans un premier temps la categorie -1
    df_balanced = (
    df_experiment.groupby("simple_target") #On limite a 6000 données
      .apply(lambda x: x.sample(n=nbr_data, random_state=42))
      .reset_index(drop=True)
      .sample(frac=1, random_state=42)  # shuffle global
      .reset_index(drop=True)           # remettre un index propre
    )
    X = []
    labels = []
    filename = []
    for idx, target in enumerate(df_balanced["id_str"]):  # On parcourt les IDs
        target_id = f"{target}"

    # Dossier à parcourir
        folder_path = os.path.join(root_data, "images")

    # Recherche dans tous les sous-dossiers
        found_image = None
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if target_id in file:
                    image_path = os.path.join(root, file)
                    found_image = image_path
                    break
            if found_image:
                break

        if found_image:
            print(f"Image trouvée : {found_image}")
            img = Image.open(found_image)
            img_array = np.array(img)   # transforme en array (H, W, C)
            X.append(img_array)
            labels.append(df_balanced["simple_target"].iloc[idx])  # associer le label
            filename.append(df_balanced["filename"].iloc[idx])
    # Construire un DataFrame avec 2 colonnes : image et label
    df_images = pd.DataFrame({
        "image": X,
        "label": labels,
        "filename": filename
    })



    return df_images



def upload_data(storage_filename, local_filename, bucket_name) :
    """_summary_
    Upload file to bucket
    Args:
        storage_filename (_type_): name of folder in the bucket where we upload our data
        local_filename (_type_): path to our folder/file
        bucket_name (_type_): our bucket name
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(storage_filename)
    blob.upload_from_filename(local_filename)
    return

def download_data(storage_filename, local_filename, bucket_name):
    """_summary_
    download file from bucket
    Args:
        storage_filename (_type_): name of folder in the bucket where we upload our data
        local_filename (_type_): path to our folder/file
        bucket_name (_type_): our bucket name
    """

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(storage_filename)
    blob.download_to_filename(local_filename)
    return
