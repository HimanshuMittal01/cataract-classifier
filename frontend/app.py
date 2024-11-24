"""Streamlit entrypoint"""

import requests

import streamlit as st
from PIL import Image

st.title("Cataract Binary Classification")

file = st.file_uploader("Upload image of an eye")

if file is not None:
    response = requests.post(
        "http://127.0.0.1:8000/predict/", files={"file": file}
    )
    st.image(Image.open(file))
    st.write(response.json())
else:
    st.warning("Please upload an image to perform prediction")
