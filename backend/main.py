from io import BytesIO

import torch
from fastapi import FastAPI, UploadFile
from PIL import Image

from cataract_classifier.predict import load_model, predict_single_image


app = FastAPI()

# Initialize model
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
model = load_model("models/finetuned_efficientnet_b0.pt")


@app.get("/")
def index():
    return {"message": "Welcome to Cataract Binary Classifier!"}


@app.post("/predict/")
async def predict(file: UploadFile, threshold: float = 0.5):
    global model

    img = Image.open(BytesIO(await file.read()))
    pred_prob = predict_single_image(model, img, device=device)

    pred_label = "Cataract" if pred_prob > threshold else "Normal"
    if pred_prob <= threshold:
        pred_prob = 1 - pred_prob

    return {"Class": pred_label, "Probability": pred_prob}
