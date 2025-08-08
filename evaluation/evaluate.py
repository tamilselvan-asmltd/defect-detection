
import os
import argparse
import yaml
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shutil

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_mlflow_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(config, mlflow_config, training_run_id):
    mlflow.set_tracking_uri(mlflow_config['mlflow_tracking_uri'])
    mlflow.set_experiment("Defect Detection Evaluation")

    processed_data_dir = config['processed_data_dir']
    registered_model_name = config['registered_model_name']
    production_reg = "production_reg"
    testing_reg = "testing_reg"
    threshold = config['threshold']
    
    output_dir = config['output_dir']

    # Get mean and std from preprocess config for normalization
    preprocess_config_path = "preprocess/config.yaml"
    preprocess_config = load_config(preprocess_config_path)
    mean = preprocess_config['mean']
    std = preprocess_config['std']
    image_size = preprocess_config['image_size']
    batch_size = config['batch_size']

    # Enable cuDNN auto-tuner for faster convolutions
    torch.backends.cudnn.benchmark = True

    # Data transformations
    data_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load test dataset
    test_dataset = datasets.ImageFolder(os.path.join(processed_data_dir, 'test'), data_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    client = mlflow.tracking.MlflowClient()
    model = None
    
    try:
        # Load the model directly from the training run artifacts
        model_uri = f"runs:/{training_run_id}/model"
        print(f"Loading model from MLflow run: {model_uri} for evaluation.")
        model = mlflow.pytorch.load_model(model_uri)
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model from MLflow run {training_run_id}: {e}")
        return

    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(test_dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"Evaluation Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    with mlflow.start_run(run_name="Evaluation Run") as run:
        mlflow.log_metrics({
            "eval_loss": avg_loss,
            "eval_accuracy": accuracy,
            "eval_precision": precision,
            "eval_recall": recall,
            "eval_f1_score": f1
        })

        # Register the model conditionally
        if accuracy >= threshold:
            print(f"Evaluation accuracy ({accuracy:.4f}) is greater than or equal to threshold ({threshold:.4f}). Registering to production registry.")
            registered_model_version = mlflow.register_model(
                model_uri=f"runs:/{training_run_id}/model",
                name=registered_model_name,
                tags={"accuracy": accuracy}
            )
            print(f"Model registered to '{registered_model_name}' with version {registered_model_version.version}.")

            # Set alias "prod" for the model with the highest accuracy in production_reg
            all_versions = client.search_model_versions(f"name='{registered_model_name}'")
            best_accuracy = -1
            best_model_version = None

            for mv in all_versions:
                if mv.tags and "accuracy" in mv.tags:
                    current_accuracy = float(mv.tags["accuracy"])
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        best_model_version = mv

            if best_model_version:
                # Remove "prod" alias from any other model versions
                for mv in all_versions:
                    aliases = client.get_model_version_aliases(name=registered_model_name, version=mv.version)
                    if "prod" in aliases and mv.version != best_model_version.version:
                        client.delete_model_version_alias(name=registered_model_name, alias="prod", version=mv.version)
                        print(f"Removed 'prod' alias from version {mv.version}.")
                
                client.set_registered_model_alias(
                    name=registered_model_name,
                    alias="prod",
                    version=best_model_version.version
                )
                print(f"Model version {best_model_version.version} of {registered_model_name} aliased as 'prod' with accuracy {best_accuracy:.4f}.")

                # Download model artifacts to evaluation_area/
                temp_download_dir = "./temp_mlflow_model_download"
                if os.path.exists(temp_download_dir):
                    shutil.rmtree(temp_download_dir)
                os.makedirs(temp_download_dir)

                downloaded_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"runs:/{training_run_id}/model",
                    dst_path=temp_download_dir
                )
                
                os.makedirs(output_dir, exist_ok=True)
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                shutil.copytree(downloaded_path, output_dir)
                print(f"Model artifacts copied to {output_dir}")
                shutil.rmtree(temp_download_dir)

            else:
                print("No model versions found to set 'prod' alias.")

        else:
            print(f"Evaluation accuracy ({accuracy:.4f}) is less than threshold ({threshold:.4f}). Registering to testing registry.")
            mlflow.register_model(
                model_uri=f"runs:/{training_run_id}/model",
                name=testing_reg,
                tags={"accuracy": accuracy}
            )
            print(f"Model registered to '{testing_reg}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a defect detection model.")
    parser.add_argument("--config", type=str, default="evaluation/config.yaml",
                        help="Path to the evaluation configuration file.")
    parser.add_argument("--mlflow_config", type=str, default="mlflow_config.yaml",
                        help="Path to the MLflow configuration file.")
    parser.add_argument("--training_run_id", type=str, required=True,
                        help="MLflow Run ID of the training run that produced the model.")
    args = parser.parse_args()

    config = load_config(args.config)
    mlflow_config = load_mlflow_config(args.mlflow_config)
    evaluate_model(config, mlflow_config, args.training_run_id)
