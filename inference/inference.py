import mlflow
import mlflow.pyfunc
import tensorflow as tf
import numpy as np
import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

IMAGE_SIZE = (128, 128)

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

if __name__ == "__main__":
    model_name = "Production_Reg"
    model_alias = "prod"

    try:
        # Load the model using the alias
        print(f"Attempting to load model '{model_name}' with alias '{model_alias}' from MLflow Model Registry...")
        loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}@{model_alias}")
        print("Model loaded successfully.")

        # Example inference
        data_dir = os.path.join(os.getcwd(), 'data')
        sample_image_path_def = os.path.join(data_dir, 'test', 'def_front', 'cast_def_0_1153.jpeg')
        sample_image_path_ok = os.path.join(data_dir, 'test', 'ok_front', 'cast_ok_0_10.jpeg')

        if os.path.exists(sample_image_path_def):
            print(f"\nRunning inference on: {sample_image_path_def}")
            input_image_def = preprocess_image(sample_image_path_def)
            prediction_def = loaded_model.predict(input_image_def)
            predicted_class_def = "Defective" if prediction_def[0][0] > 0.5 else "OK"
            print(f"Prediction for defective image: {prediction_def[0][0]:.4f} -> {predicted_class_def}")
        else:
            print(f"Sample defective image not found at {sample_image_path_def}")

        if os.path.exists(sample_image_path_ok):
            print(f"\nRunning inference on: {sample_image_path_ok}")
            input_image_ok = preprocess_image(sample_image_path_ok)
            prediction_ok = loaded_model.predict(input_image_ok)
            predicted_class_ok = "Defective" if prediction_ok[0][0] > 0.5 else "OK"
            print(f"Prediction for OK image: {prediction_ok[0][0]:.4f} -> {predicted_class_ok}")
        else:
            print(f"Sample OK image not found at {sample_image_path_ok}")

    except Exception as e:
        print(f"Error loading model or performing inference: {e}")
        print("Please ensure a model is registered in MLflow Model Registry under 'Production_Reg' with alias 'prod'.")
