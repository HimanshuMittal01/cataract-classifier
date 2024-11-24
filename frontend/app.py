import streamlit as st
import requests

st.title("Iris Species Prediction")

file = st.file_uploader("Upload image of an eye")

response = requests.post("http://127.0.0.1:8000/predict/", files={"file": file})
st.write(response.json())
