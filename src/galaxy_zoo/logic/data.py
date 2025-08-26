import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

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
