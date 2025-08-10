import mlflow
import mlflow.keras
import tensorflow as tf
import yaml
import os
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score

# Load re_train config
with open("re_train/re_train.yaml", "r") as f:
    re_train_config = yaml.safe_load(f)

# Load main config
with open("config.yaml", "r") as f:
    main_config = yaml.safe_load(f)

config = {**main_config, **re_train_config} # Merge configs, re_train_config overrides main_config for common keys

data_path = config["paths"]["data_path"]
image_size = tuple(config["model_parameters"]["image_size"])
batch_size = config["model_parameters"]["batch_size"]
epochs = config["model_parameters"]["epochs"]
learning_rate = config["model_parameters"]["learning_rate"]

experiment_name = config["mlflow"]["experiment_name"]
testing_model_name = config["mlflow"]["testing_model_name"]
testing_model_alias = config["mlflow"]["testing_model_alias"]

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Set MLflow experiment
mlflow.set_experiment(experiment_name)

# Load latest model from MLflow using alias
model_uri = f"models:/{testing_model_name}@{testing_model_alias}"
print(f"Loading model from: {model_uri}")
model = mlflow.keras.load_model(model_uri)

# Prepare datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_path, "train"),
    image_size=image_size,
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_path, "test"),
    image_size=image_size,
    batch_size=batch_size
)

# Compile model for re-training
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Start MLflow run
with mlflow.start_run():
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Log the retrained model
    mlflow.keras.log_model(model, artifact_path="model")

    # Log metrics
    final_acc = history.history["val_accuracy"][-1]
    mlflow.log_metric("val_accuracy", final_acc)

    # --- Evaluation and Registration Logic (from src/evaluation.py) ---
    client = MlflowClient()

    test_dir = os.path.join(config['paths']['data_path'], 'test')

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='binary',
        image_size=image_size,
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=False
    )

    print("Evaluating model on test dataset...")
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred = (y_pred_probs > 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    mlflow.log_metric("test_accuracy", accuracy)

    model_name = config['mlflow']['production_model_name'] if accuracy >= config['mlflow']['evaluation_threshold'] else config['mlflow']['testing_model_name']
    print(f"Model name chosen for registration: {model_name}")

    # Log the retrained model
    mlflow.keras.log_model(model, artifact_path="model")
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags={"acc": str(accuracy)}
    )
    print(f"Model registered as {model_name} version {registered_model.version}")

    # Manage aliases for Production_Reg
    if model_name == config['mlflow']['production_model_name']:
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

print("Retraining complete and model logged to MLflow.")

# Clear the Keras session to free up resources
tf.keras.backend.clear_session()
