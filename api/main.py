from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import keras

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model using Keras 3 TFSMLayer
MODEL = keras.layers.TFSMLayer(r"../models/1", call_endpoint="serving_default")

# ⚠️ Update this list to match your training classes
CLASS_NAMES = [
    "Acne and Rosacea",
    "Basal Cell Carcinoma",
    "Hair Loss (Alopecia)",
    "Ringworm"
]

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((224, 224))  # 👈 change if your model input size is different
    image = np.array(image).astype("float32") / 255.0  # normalize to [0,1]
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    # ✅ Call model (returns dict in Keras 3)
    outputs = MODEL(img_batch)

    # ✅ Extract tensor from dict
    predictions = list(outputs.values())[0].numpy()

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        "class": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
