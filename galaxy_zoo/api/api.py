from galaxy_zoo.logic.registry import load_model
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras

app = FastAPI()
#model = keras.models.load_model("models/model_tests/model_VGG16_NM_424_200.keras")


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
    return {"resp":"Hello world"}
    # $CHA_END

#uvicorn galaxy_zoo.api.api:app --reload

#uvicorn galaxy_zoo.api.api:app --reload
