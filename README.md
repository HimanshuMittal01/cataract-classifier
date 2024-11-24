# Cataract Binary Classification

![App Demo Screenshot](docs/static/sample_app_ss.png)

This project provides a machine learning-based solution for classifying eye images to determine if the eye shows signs of cataracts. The model uses a fine-tuned **EfficientNet-B0** architecture to predict whether an image belongs to the "Cataract" or "Normal" class. 

The system consists of three parts:
1. **Backend API**: A FastAPI server for handling image uploads and predictions.
2. **Frontend App**: A Streamlit app for users to upload images and view the prediction results.
3. **Model Training**: Source code for fine-tuning models.

## Setup Environment



## Launch App

Streamlit frontend calls API written in FastAPI. So, we need to start both the services.

**1. Start the backend server:**

- Using `uv`
    ```zsh
    uv run fastapi dev backend/main.py
    ```

- Otherwise
    ```zsh
    fastapi dev backend/main.py

    # Use `fastapi run backend/main.py` for production
    ```

Verify that API is running successfully at http://127.0.0.1:8000/ </br>
View API docs at http://127.0.0.1:8000/docs

**2. Start the frontend server:**

```zsh
streamlit run frontend/app.py
```

This will open http://localhost:8501/ in your browser where you can upload images and it will predict cataract or not cataract.

## How to Train Model

Simply run
```
python main.py
```

There are other options you can specify like `num-epochs`. See all options using `python main.py --help`
```
python main.py --num-epochs 4
```

### Potential Improvements:
- [ ] Setup tensorboard for monitoring
- [ ] Add more metrics - precision, recall, f1-score, classification report, confusion matrix, ROC curve
- [ ] Write API usage documentation with example requests and responses
- [ ] Write a CNN model from scratch
- [ ] [Optional] Provide three model options in the frontend
