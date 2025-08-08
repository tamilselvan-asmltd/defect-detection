import tensorflow as tf
import mlflow
import mlflow.tensorflow
from mlflow.entities import ViewType
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Defect_Detection_CNN_Training")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5 # Reduced for faster execution in CI/CD, can be increased

def create_cnn_model(input_shape, num_filters, kernel_size, activation='relu'):
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=input_shape),
        tf.keras.layers.Conv2D(num_filters, kernel_size, activation=activation),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(num_filters * 2, kernel_size, activation=activation),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=activation),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), 'data')
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test') # Used for validation in this script

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='binary',
        image_size=IMAGE_SIZE,
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, # Using test data as validation for simplicity in this example
        labels='inferred',
        label_mode='binary',
        image_size=IMAGE_SIZE,
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        shuffle=False # No need to shuffle validation data
    )

    # Hyperparameter sets to experiment with
    hyperparameter_sets = [
        {"learning_rate": 0.001, "num_filters": 16, "kernel_size": (3, 3)},
        {"learning_rate": 0.0005, "num_filters": 32, "kernel_size": (3, 3)},
        {"learning_rate": 0.001, "num_filters": 32, "kernel_size": (5, 5)},
    ]

    best_val_accuracy = -1
    best_run_id = None
    best_model_path = None

    for i, hparams in enumerate(hyperparameter_sets):
        with mlflow.start_run(run_name=f"CNN_Run_{i+1}") as run:
            mlflow.tensorflow.autolog() # Autologs metrics, parameters, and models

            print(f"Starting run with hyperparameters: {hparams}")
            mlflow.log_params(hparams)

            model = create_cnn_model(
                input_shape=IMAGE_SIZE + (3,),
                num_filters=hparams["num_filters"],
                kernel_size=hparams["kernel_size"]
            )

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=hparams["learning_rate"]),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy']
            )

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS
            )

            # Manually log the final validation accuracy if autologging doesn't capture it directly
            final_val_accuracy = history.history['val_accuracy'][-1]
            mlflow.log_metric("final_val_accuracy", final_val_accuracy)

            # Check if this is the best model
            if final_val_accuracy > best_val_accuracy:
                best_val_accuracy = final_val_accuracy
                best_run_id = run.info.run_id
                
                # Save the best model locally
                temp_model_path = "temp_best_model"
                if not os.path.exists(temp_model_path):
                    os.makedirs(temp_model_path)
                
                model.save(os.path.join(temp_model_path, "best_cnn_model.keras"))
                best_model_path = os.path.join(temp_model_path, "best_cnn_model.keras")
                print(f"New best model found with validation accuracy: {best_val_accuracy:.4f}")
    
    print(f"Training complete. Best model (Run ID: {best_run_id}) had validation accuracy: {best_val_accuracy:.4f}")

    # Save the best run ID and model path for evaluation.py
    with open("best_model_info.txt", "w") as f:
        f.write(f"run_id:{best_run_id}\n")
        f.write(f"model_path:{best_model_path}\n")
    print("Best model info saved to best_model_info.txt")
