import mlflow
import mlflow.pyfunc
import tensorflow as tf
import numpy as np
import os
import yaml
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

IMAGE_SIZE = tuple(config['model_parameters']['image_size'])


def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

if __name__ == "__main__":
    model_name = config['mlflow']['inference_model_name']
    model_alias = config['mlflow']['inference_model_alias']

    try:
        # Load the model using the alias
        print(f"Attempting to load model '{model_name}' with alias '{model_alias}' from MLflow Model Registry...")
        model_uri = f"models:/{model_name}@{model_alias}"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        model_version = mlflow.tracking.MlflowClient().get_model_version_by_alias(model_name, model_alias)
        print("Model loaded successfully.")
        print(f"Loaded model version: {model_version.version}")

        # Example inference
        data_dir = os.path.join(os.getcwd(), config['paths']['data_path'])
        sample_image_path_def = os.path.join(data_dir, config['inference_examples']['defective_image'])
        sample_image_path_ok = os.path.join(data_dir, config['inference_examples']['ok_image'])

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
