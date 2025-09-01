from galaxy_zoo.logic.registry import load_model
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from PIL import Image
import tensorflow as tf


app = FastAPI()

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


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("In /predict")
    # Lire le contenu du fichier
    content = await file.read()


    # image = Image.open(tf.io.BytesIO(content))
    # print("image content \n", image)

    result = {"class": "Spiral", "confidence": 0.93}

    return JSONResponse(result)
