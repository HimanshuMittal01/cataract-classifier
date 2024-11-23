from fastapi import FastAPI, UploadFile

app = FastAPI()


@app.get("/")
def hello():
    return {"message": "Hello World"}


@app.post("/predict/")
async def predict(file: UploadFile):
    # TODO: Use CNN to predict class
    pred_prob = 0.8
    pred_label = "Cataract" if pred_prob > 0.7 else "Normal"

    return {"Class": pred_label, "Probability": pred_prob}
