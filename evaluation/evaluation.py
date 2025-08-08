import tensorflow as tf
import mlflow
from mlflow.tracking import MlflowClient
import os
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

if __name__ == "__main__":
    # Read best model info from file
    best_run_id = None
    best_model_local_path = None
    try:
        with open("best_model_info.txt", "r") as f:
            for line in f:
                if line.startswith("run_id:"):
                    best_run_id = line.split(":")[1].strip()
                elif line.startswith("model_path:"):
                    best_model_local_path = line.split(":")[1].strip()
    except FileNotFoundError:
        print("Error: best_model_info.txt not found. Run train.py first.")
        exit()

    if not best_run_id or not best_model_local_path:
        print("Error: Could not retrieve best run ID or model path from best_model_info.txt.")
        exit()

    print(f"Loading best model from: {best_model_local_path}")
    loaded_model = tf.keras.models.load_model(best_model_local_path)

    data_dir = os.path.join(os.getcwd(), 'data')
    test_dir = os.path.join(data_dir, 'test')

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='binary',
        image_size=IMAGE_SIZE,
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Evaluating model on test dataset...")
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred_probs = loaded_model.predict(test_ds)
    y_pred = (y_pred_probs > 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Register model in MLflow Model Registry
    client = MlflowClient()
    model_name = "Production_Reg" if accuracy >= 0.60 else "Testing_Reg"

    with mlflow.start_run(run_id=best_run_id) as run:
        # Log evaluation metrics to the same run as training
        mlflow.log_metric("test_accuracy", accuracy)

        # Register the model
        model_uri = f"runs:/{best_run_id}/model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags={"acc": str(accuracy)}
        )
        print(f"Model registered as {model_name} version {registered_model.version}")

    # Manage aliases for Production_Reg
    if model_name == "Production_Reg":
        print(f"Managing aliases for {model_name}...")
        latest_versions = client.search_model_versions(f"name='{model_name}'")

        best_accuracy_in_registry = -1.0
        best_version_for_prod_alias = None

        for mv in latest_versions:
            if "acc" in mv.tags:
                current_accuracy = float(mv.tags["acc"])
                if current_accuracy > best_accuracy_in_registry:
                    best_accuracy_in_registry = current_accuracy
                    best_version_for_prod_alias = mv.version

        if best_version_for_prod_alias:
            # Remove existing 'prod' alias if any
            for mv in latest_versions:
                if "prod" in mv.aliases:
                    print(f"Removing existing 'prod' alias from version {mv.version}")
                    client.delete_registered_model_alias(name=model_name, alias="prod")
            
            # Set 'prod' alias to the best version
            client.set_registered_model_alias(name=model_name, alias="prod", version=best_version_for_prod_alias)
            print(f"Set 'prod' alias for {model_name} to version {best_version_for_prod_alias} with accuracy {best_accuracy_in_registry:.4f}")
        else:
            print("No versions found with 'acc' tag for Production_Reg to set alias.")
