from galaxy_zoo.logic.registry import load_model
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from galaxy_zoo.logic.params import *

app = FastAPI()


TARGET_NAMES = {
    0: "Elliptical",
    1: "Spiral",
    2: "Edge-on / Cigar",
    -1: "Other"
}
TARGET_NAMES_6 = {
    0: "Elliptical",
    1: "Round Elliptical",
    2: "Cigar",
    3: "Edge-on Disk",
    4: "Spiral",
    5: "Barred Spiral",
    6: "No Bar Or Spiral",
}


def preprocess_bytes(image_bytes: bytes, size=(256, 256)) -> tf.Tensor:
    """Decode bytes -> RGB, resize, float32 [0,1], ajoute la dimension batch."""
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)  # (H,W,3), dtype=uint8
    img = tf.image.resize(img, size)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.expand_dims(img, 0)  # (1,H,W,3)
    return img



# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    # $CHA_BEGIN
    return "Hello world"
    # $CHA_END

app.state.modelVGG = load_model("20250903-113623VGG16.h5")
app.state.model6 = load_model("20250902-125159.h5")
app.state.modelCNN = load_model("20250903-105248CNN.h5")
app.state.modelCNN7 = load_model("20250903-234811_7_CAT_MODEL_SMALL_NICOLAS_256-256X5250.h5")

@app.post("/predictVGG")
async def predictVGG(file: UploadFile = File(...)):
    # Vérif MIME
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Please upload a JPEG or PNG image.")
    model = app.state.modelVGG
    # Lire et prétraiter
    contents = await file.read()
    try:
        img = preprocess_bytes(contents, size=(256,256))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Prédire
    pred = model.predict(img)
    cls_id = int(np.argmax(pred, axis=1)[0])# (1, num_classes)
    proba  = float(np.max(pred, axis=1)[0])


    return {
        "predicted_class": TARGET_NAMES.get(cls_id, "Other"),
        "probability": proba,

    }


@app.post("/predict6")
async def predict6(file: UploadFile = File(...)):
    # Vérif MIME
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Please upload a JPEG or PNG image.")
    model = app.state.model6
    # Lire et prétraiter
    contents = await file.read()
    try:
        img = preprocess_bytes(contents, size=(224,224))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Prédire
    pred = model.predict(img)
    cls_id = int(np.argmax(pred, axis=1)[0])# (1, num_classes)
    proba  = float(np.max(pred, axis=1)[0])


    return {
        "predicted_class": TARGET_NAMES_6.get(cls_id, "Other"),
        "probability": proba,

    }

@app.post("/predictCNN")
async def predictCNN(file: UploadFile = File(...)):
    # Vérif MIME
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Please upload a JPEG or PNG image.")
    model = app.state.modelCNN
    # Lire et prétraiter
    contents = await file.read()
    try:
        img = preprocess_bytes(contents, size=(256,256))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Prédire
    pred = model.predict(img)
    cls_id = int(np.argmax(pred, axis=1)[0])# (1, num_classes)
    proba  = float(np.max(pred, axis=1)[0])


    return {
        "predicted_class": TARGET_NAMES.get(cls_id, "Other"),
        "probability": proba,

    }

@app.post("/predictCNN7")
async def predictCNN7(file: UploadFile = File(...)):
    # Vérif MIME
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Please upload a JPEG or PNG image.")
    model = app.state.modelCNN7
    # Lire et prétraiter
    contents = await file.read()
    try:
        img = preprocess_bytes(contents, size=(256,256))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Prédire
    pred = model.predict(img)
    cls_id = int(np.argmax(pred, axis=1)[0])# (1, num_classes)
    proba  = float(np.max(pred, axis=1)[0])


    return {
        "predicted_class": TARGET_NAMES_7.get(cls_id, "Other"),
        "probability": proba,

    }
