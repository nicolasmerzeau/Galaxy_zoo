from galaxy_datasets import gz2
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, Any
import os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

RANDOM_STATE = 42
CURRENT_DIR = os.path.dirname(__file__) # Garde le chemin vers data.py peut import où la fonction est appelé

ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../raw_data")) # on revient a src puis galaxy_zoo
FILE_PATH = os.path.join(ROOT, "gz2_train_catalog.parquet")
LABEL_MAP = {
    0:0,
    1:0,
    2:2,
    3:2,
    4:1,
    5:1,
    6:-1,
    -1: -1
}
# target_names = {
#     0: "Elliptical",
#     1: "Spiral",
#     2: "Edge-on / Cigar",
#     -1: "Other"
# }

def generate_image_df(nb_data = 2000, target_class = -1) :
    """
    Generates a balanced DataFrame containing image file paths and target labels for a classification experiment.

    This function loads experiment data from a parquet file, maps labels to simplified targets, balances the dataset
    by sampling up to `nb_data` samples per class (or `nb_data * 2` for the target class if specified), and constructs
    file paths for each image based on its filename prefix.

    Args:
        nb_data (int, optional): Number of samples to include per class. If the class matches `target_class`,
            up to `nb_data * 2` samples are included. Defaults to 2000.
        target_class (int, optional): The class to oversample (include up to `nb_data * 2` samples).
            If set to -1, no class is oversampled. Defaults to -1.

    Returns:
        pandas.DataFrame: A DataFrame with columns:
            - 'prefix': The first 6 characters of the filename, used as a subfolder name.
            - 'filename': The image filename.
            - 'path': The full path to the image file.
            - 'simple_target': The mapped target label for classification.
    """
    # Load data updated returns img path instead of matrices


    # Lecture
    df_experiment = pd.read_parquet(FILE_PATH)

    df_experiment["simple_target"] = df_experiment["label"].map(LABEL_MAP)
    df_experiment = df_experiment[df_experiment["simple_target"] != -1]
    df_balanced = ( # on équilibre le dataset (if classe cible -> nb_data * 2)
        df_experiment.groupby("simple_target")
        .apply(lambda x: x.sample(
            n=min(nb_data * 2 if x['simple_target'].iloc[0] == target_class else nb_data, len(x)),
            random_state=RANDOM_STATE
        ))
        .reset_index(drop=True)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    # Dossier à parcourir
    folder_path = os.path.join(ROOT, "images")

    # Recherche du sous dossier à partir des 6 premiers caractères du filename
    df_balanced['prefix'] = df_balanced['filename'].str.slice(0, 6)
    df_balanced['path'] = df_balanced.apply(lambda r: os.path.join(folder_path, r['prefix'], r['filename']), axis=1)

    return df_balanced[['prefix', 'filename', 'path', 'simple_target' ]]


def load_and_preprocess_data(df: pd.DataFrame,
                            # nb_data: int = 2000,
                            ovr: bool = True,
                            target_class: int = 0,
                            target_size: Tuple[int, int] = (128, 128),
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
        Loads and preprocesses images from a DataFrame, resizing them and creating binary labels for a specified target class.
        Args:
            df (pd.DataFrame): DataFrame containing image file paths in the 'path' column and class labels in the 'simple_target' column.
            target_class (int, optional): The class to be considered as positive (1) in the binary classification. Defaults to 0.
            target_size (Tuple[int, int], optional): Desired size (height, width) to resize images to. Defaults to (224, 224).
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - X: Array of preprocessed images of shape (num_images, height, width, channels).
                - y: Array of binary labels (1 for target_class, 0 for others).
    """
    images = []
    labels = []

    # df = generate_image_df(nb_data)
    # print(f"Chargement de {len(df)} images...")

    for idx, row in df.iterrows():
        # Charger l'image
        image_path = row['path']

        img = tf.io.read_file(image_path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, target_size)
        img = tf.image.convert_image_dtype(img, tf.float32)

        images.append(img)
        original_label = int(row['simple_target'])
        if ovr:
            # Créer le label binary One vs Rest
            binary_label = 1 if original_label == target_class else 0
            labels.append(binary_label)
        else:
            labels.append(original_label)

    X = np.array(images)
    y = np.array(labels)

    print(f"{len(X)} images chargées avec succès")
    print(f"   Shape des images: {X.shape}")

    return X, y


def load_data() :


    current_dir = os.path.dirname(__file__) #Garde le chemin vers data.py peut import où la fonction est appelé

    root = os.path.abspath(os.path.join(current_dir, "../../raw_data")) # on revient a src puis galaxy_zoo

    train_catalog, label_columns = gz2(
    root=root,
    download=True,
    train=True
    )
    # Chemin vers ton fichier
    file_path = os.path.join(root, "gz2_train_catalog.parquet")

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
      .apply(lambda x: x.sample(n=2000, random_state=42))
      .reset_index(drop=True)
      .sample(frac=1, random_state=42)  # shuffle global
      .reset_index(drop=True)           # remettre un index propre
    )
    X = []
    for target in df_balanced["id_str"] : #On crée le dataframe d'image

        target_id = f"{target}"

        # Dossier à parcourir
        folder_path = os.path.join(root, "images") # ← modifie ici si besoin

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
            X.append(img)
    X = pd.DataFrame(X)
    y = df_balanced["simple_target"]




    return (X,y)
