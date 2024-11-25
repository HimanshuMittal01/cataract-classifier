"""Backend entrypoint for the Cataract Binary Classification API.

This FastAPI app allows for image uploads to classify whether an eye image shows signs of cataracts.
The model is a fine-tuned EfficientNet-B0 architecture that outputs a probability score, which is used
to classify the image as "Cataract" or "Normal".

The model is loaded at startup, and predictions can be made via the `/predict/` endpoint.
"""

from io import BytesIO

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, HTTPException, Query, File

from cataract_classifier.predict import load_model, predict_single_image


app = FastAPI()

# Check for available hardware (GPU/CPU) and set device accordingly
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Load the pre-trained model (ensure that the model file exists at the specified path)
try:
    model_name = "efficientnet_b0"
    model = load_model(
        model_name, f"results/{model_name}/finetuned_{model_name}.pt"
    )
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")


@app.get("/")
def index():
    """Root endpoint to check if the API is working."""
    return {"message": "Welcome to Cataract Binary Classifier!"}


@app.post(
    "/predict/",
    response_description="The predicted class label and the associated probability for the uploaded image.",
    responses={
        200: {
            "description": "Successful prediction with class and probability.",
            "content": {
                "application/json": {
                    "example": {"Class": "Cataract", "Probability": 0.85}
                }
            },
        },
        400: {
            "description": "Invalid image file uploaded.",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid image file."}
                }
            },
        },
        500: {
            "description": "Prediction error from the model.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Prediction error: model error message"
                    }
                }
            },
        },
    },
)
async def predict(
    file: UploadFile = File(
        ...,
        description="The image file to be uploaded for prediction. Supported formats include PNG, JPG, JPEG.",
    ),
    threshold: float = Query(
        0.5,
        ge=0,
        le=1,
        description="The probability threshold for classification. If the model's probability is greater than or equal to this threshold, the image is classified as 'Cataract'. Otherwise, it is classified as 'Normal'.",
    ),
):
    """Predict whether a given image contains cataracts or not."""
    global model

    # Validate the file type - check if the uploaded file is an image
    try:
        img = Image.open(BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Use the model to predict the probability of cataract presence
    try:
        pred_prob = predict_single_image(model, img, device=device)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}"
        )

    # Create response
    pred_label = "Cataract" if pred_prob > threshold else "Normal"
    if pred_prob <= threshold:
        pred_prob = 1 - pred_prob

    return {"Class": pred_label, "Probability": pred_prob}
