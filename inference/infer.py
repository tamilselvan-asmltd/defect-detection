
import os
import argparse
import yaml
import torch
from torchvision import transforms
from PIL import Image
import mlflow.pytorch

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_mlflow_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def infer_image(config, mlflow_config):
    mlflow.set_tracking_uri(mlflow_config['mlflow_tracking_uri'])

    model_name = config['model_name']
    model_alias = config['model_alias']
    image_path = config['image_path']

    if not image_path or not os.path.exists(image_path):
        print(f"Error: Image path '{image_path}' is not valid or does not exist.")
        return

    # Get mean and std from preprocess config for normalization
    preprocess_config_path = "preprocess/config.yaml"
    preprocess_config = load_config(preprocess_config_path)
    mean = preprocess_config['mean']
    std = preprocess_config['std']
    image_size = preprocess_config['image_size']

    # Data transformations for inference
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        # Load the model from MLflow Model Registry
        model_uri = f"models:/{model_name}@{model_alias}"
        model = mlflow.pytorch.load_model(model_uri)
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        print(f"Please ensure a model named '{model_name}' with alias '{model_alias}' exists in MLflow Model Registry.")
        return

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0) # Add batch dimension
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_class = torch.max(outputs, 1)

    # Assuming class 0 is 'def_front' and class 1 is 'ok_front' based on ImageFolder sorting
    class_names = ['def_front', 'ok_front'] # This should ideally be loaded from training artifacts
    predicted_label = class_names[predicted_class.item()]
    confidence = probabilities[0][predicted_class.item()].item()

    print(f"Inference Result for {image_path}:")
    print(f"  Predicted Class: {predicted_label}")
    print(f"  Confidence: {confidence:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--config", type=str, default="inference/config.yaml",
                        help="Path to the inference configuration file.")
    parser.add_argument("--mlflow_config", type=str, default="mlflow_config.yaml",
                        help="Path to the MLflow configuration file.")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to the image for inference.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.image_path:
        config['image_path'] = args.image_path

    mlflow_config = load_mlflow_config(args.mlflow_config)
    infer_image(config, mlflow_config)
