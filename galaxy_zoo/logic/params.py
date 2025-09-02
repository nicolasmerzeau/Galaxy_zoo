import os
import numpy as np

##################  VARIABLES  ##################

MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")
GAR_IMAGE = os.environ.get("GAR_IMAGE")
GAR_MEMORY = os.environ.get("GAR_MEMORY")


##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "nicolasmerzeau", "Galaxy_zoo", "raw_data")
RANDOM_STATE = 42
IMG_SIZE = 256
INPUT_SHAPE = (256,256, 3)
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
LABEL_MAP_7 = {
    0:0,
    1:1,
    2:2,
    3:3,
    4:4,
    5:5,
    6:6,
    -1: -1
}

#### Nicolas a modifié le dictionnaire pour une classification en 6 catégories #####

CURRENT_DIR = os.path.dirname(__file__) # Garde le chemin vers params.pu peut import où la fonction est appelé

if MODEL_TARGET == "local" :
    ROOT_DATA = os.path.abspath(os.path.join(CURRENT_DIR, "../../raw_data")) # on revient a src puis galaxy_zoo
else :
    ROOT_DATA = "/raw_data"
ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../")) # on revient a GALAXY_ZOO
LOCAL_REGISTRY_PATH =  os.path.abspath(os.path.join(ROOT, "training_outputs"))
