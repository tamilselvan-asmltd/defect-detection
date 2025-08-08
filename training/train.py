
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_mlflow_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(config, mlflow_config):
    mlflow.set_tracking_uri(mlflow_config['mlflow_tracking_uri'])
    mlflow.set_experiment("Defect Detection Training")

    with mlflow.start_run() as run:
        print(f"mlflow_run_id: {run.info.run_id}")
        # Log hyperparameters
        mlflow.log_params({
            "model_name": config['model_name'],
            "batch_size": config['batch_size'],
            "epochs": config['epochs'],
            "learning_rate": config['learning_rate'],
            "num_classes": config['num_classes']
        })

        processed_data_dir = config['processed_data_dir']
        image_size = 224 # Hardcoded for ResNet, should match preprocess config
        # Get mean and std from preprocess config for normalization
        preprocess_config_path = "preprocess/config.yaml"
        preprocess_config = load_config(preprocess_config_path)
        mean = preprocess_config['mean']
        std = preprocess_config['std']

        # Data transformations (including normalization)
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
        }

        # Load datasets
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(processed_data_dir, x),
                                    data_transforms[x])
            for x in ['train', 'test']
        }

        # Class balancing for training data
        train_labels = [label for _, label in image_datasets['train'].samples]
        class_counts = torch.tensor([train_labels.count(i) for i in range(config['num_classes'])])
        class_weights = 1. / class_counts.float()
        sample_weights = torch.tensor([class_weights[t] for t in train_labels])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=config['batch_size'],
                                  sampler=sampler, num_workers=4),
            'test': DataLoader(image_datasets['test'], batch_size=config['batch_size'],
                                 shuffle=False, num_workers=4) # No shuffle for test
        }

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load pre-trained ResNet18 model
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, config['num_classes'])
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

        best_accuracy = 0.0

        for epoch in range(config['epochs']):
            model.train()
            running_loss = 0.0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders['train']:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(image_datasets['train'])
            epoch_accuracy = accuracy_score(all_labels, all_preds)

            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_accuracy, step=epoch)

            print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}")

            # Evaluate on test set after each epoch (or periodically)
            model.eval()
            test_running_loss = 0.0
            test_all_preds = []
            test_all_labels = []
            with torch.no_grad():
                for inputs, labels in dataloaders['test']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    test_all_preds.extend(preds.cpu().numpy())
                    test_all_labels.extend(labels.cpu().numpy())
            
            test_epoch_loss = test_running_loss / len(image_datasets['test'])
            test_epoch_accuracy = accuracy_score(test_all_labels, test_all_preds)
            mlflow.log_metric("test_loss", test_epoch_loss, step=epoch)
            mlflow.log_metric("test_accuracy", test_epoch_accuracy, step=epoch)
            print(f"Test Loss: {test_epoch_loss:.4f} Test Acc: {test_epoch_accuracy:.4f}")

            # Save best model based on test accuracy
            if test_epoch_accuracy > best_accuracy:
                best_accuracy = test_epoch_accuracy
                # Log the best model to MLflow
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="model"
                )
                print(f"Current best test accuracy ({best_accuracy:.4f}) is greater than previous best. Logging model to MLflow.")
                print(f"Model logged to MLflow with test accuracy: {best_accuracy:.4f}")

                

        mlflow.log_metric("final_best_test_accuracy", best_accuracy)
        print(f"Training complete. Best Test Accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a defect detection model.")
    parser.add_argument("--config", type=str, default="training/config.yaml",
                        help="Path to the training configuration file.")
    parser.add_argument("--mlflow_config", type=str, default="mlflow_config.yaml",
                        help="Path to the MLflow configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)
    mlflow_config = load_mlflow_config(args.mlflow_config)
    train_model(config, mlflow_config)
