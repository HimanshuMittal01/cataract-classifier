"""Frontend entrypoint for Cataract Binary Classification.

This Streamlit app allows users to upload an image of an eye and sends the image to a backend
API for prediction on whether the eye is affected by cataract or not.
The backend API is expected to be running at http://127.0.0.1:8000/predict/.

Supported image formats: PNG, JPG, JPEG.
"""

import requests

import streamlit as st
from PIL import Image

# Set the title of the app
st.title("Cataract Binary Classification")

# Allow only PNG, JPG, or JPEG
file = st.file_uploader("Upload image of an eye", type=["png", "jpg", "jpeg"])

if file is not None:
    try:
        # Send the image to the prediction API (assumed to be running locally)
        response = requests.post(
            "http://127.0.0.1:8000/predict/", files={"file": file}
        )

        # If the response from the API is successful, display the image and the prediction result
        if response.status_code == 200:
            col1, col2, col3 = st.columns(3)
            with col2:
                st.image(Image.open(file), caption="Uploaded Image", width=224)
                response_json = response.json()
                st.success(
                    f"Prediction: **{response_json['Class']}**\nProbability: **{response_json['Probability']:.4f}**"
                )
        else:
            st.error(
                "Error in prediction. Status code: {}".format(
                    response.status_code
                )
            )

    except Exception as e:
        st.error(f"An error occurred while communicating with the backend: {e}")

else:
    st.warning(
        "Please upload an image. Only 'png', 'jpg', 'jpeg' formats are supported."
    )
