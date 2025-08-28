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
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "code", "nicolasmerzeau", "Galaxy_zoo", "training_outputs")
RANDOM_STATE = 42
IMG_SIZE = 256
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
