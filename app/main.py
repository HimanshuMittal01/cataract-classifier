from io import BytesIO

from fastapi import FastAPI, UploadFile
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

app = FastAPI()


@app.get("/")
def hello():
    return {"message": "Hello World"}


@app.post("/predict/")
async def predict(file: UploadFile):
    # TODO: Use CNN to predict class
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)
    model.load_state_dict(
        torch.load("models/finetuned_efficientnet_b0.pt", weights_only=True)
    )
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )

    img = transform(Image.open(BytesIO(await file.read())))
    prediction = model(img.unsqueeze(0))

    pred_prob = torch.sigmoid(prediction).item()
    pred_label = "Cataract" if pred_prob > 0.7 else "Normal"

    return {"Class": pred_label, "Probability": pred_prob}
