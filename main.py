import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from PIL import Image
import numpy as np
import io
import json

app = FastAPI()

# Allow frontend localhost calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------- Load Models & Files ---------- #

MODEL_DIR = "models"

# Load tokenizer JSON
with open(f"{MODEL_DIR}/tokenizer.json", "r", encoding="utf-8") as f:
    json_str = f.read()
    tokenizer = tokenizer_from_json(json_str)


# Load max_length
with open(f"{MODEL_DIR}/max_length.txt", "r") as f:
    max_length = int(f.read().strip())

# Load vocab_size
with open(f"{MODEL_DIR}/vocab_size.txt", "r") as f:
    vocab_size = int(f.read().strip())

# Load trained caption model (.h5)
model = load_model(f"{MODEL_DIR}/caption_model_best.h5")

# Load VGG16 encoder
vgg = VGG16()
vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)


# ------------ Helper Functions ------------ #

def extract_features(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))             # resize
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = vgg.predict(image, verbose=0)
    return feature


def idx_to_word(integer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(feature):
    in_text = "startseq"

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat)

        if word is None:
            break
        in_text += " " + word

        if word == "endseq":
            break

    return in_text.replace("startseq", "").replace("endseq", "").strip()


# --------------- API ENDPOINTS -------------- #

@app.post("/predict-caption")
async def caption_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    feature = extract_features(image_bytes)
    result = predict_caption(feature)
    return {"caption": result}


@app.get("/")
def home():
    return {"status": "Image Captioning API Running!"}


# --------------- Start Server --------------- #

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
