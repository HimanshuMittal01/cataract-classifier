"""Module for predicting cataract on images.

It is intended that it will be used by backend.
"""

import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def load_model():
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)
    model.load_state_dict(
        torch.load("models/finetuned_efficientnet_b0.pt", weights_only=True)
    )
    return model


def predict_single_image(model, image):
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    img = transform(image)
    prediction = model(img.unsqueeze(0))

    pred_prob = torch.sigmoid(prediction).item()
    return pred_prob
