# Defect Detection Pipeline with PyTorch and MLflow

This project implements an end-to-end defect detection pipeline using PyTorch for model training and MLflow for experiment tracking, model management, and deployment. The pipeline automates the process from data preprocessing to model inference, with CI/CD capabilities via GitHub Actions.

## Features

*   **Data Preprocessing**: Resizes images, normalizes them, and handles class balancing.
*   **Model Training**: Trains a CNN model (ResNet18) using PyTorch, with comprehensive MLflow tracking of metrics, parameters, and artifacts.
*   **Model Evaluation**: Evaluates the trained model on a test set, logs metrics to MLflow, and promotes the model to a 'prod' alias in the MLflow Model Registry if performance criteria are met.
*   **Model Inference**: Loads the 'prod' aliased model from the MLflow Model Registry to perform predictions on new images.
*   **MLflow Integration**: Utilizes MLflow for experiment tracking, model versioning, and model registry.
*   **CI/CD Automation**: GitHub Actions workflow to automate the entire pipeline on pushes to the `main` branch.

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.8+**
*   **pip** (Python package installer)
*   **Git** (for cloning the repository)
*   **MLflow** (will be installed via `requirements.txt`, but ensure your environment is clean)

## Setup

Follow these steps to set up the project locally:

1.  **Clone the Repository**:
    ```bash
    git clone <your-repository-url>
    cd casting_v2 # Or wherever your project root is
    ```
    *(Assuming your project is in a Git repository. If not, navigate to your project directory.)*

2.  **Install Dependencies**:
    Navigate to the project root directory and install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    It's highly recommended to use a virtual environment (e.g., `conda` or `venv`) to manage dependencies.

3.  **Start the MLflow Tracking Server**:
    The MLflow Tracking Server is essential for logging experiments, models, and managing the model registry. Run the following command in your terminal from the project root. This will start a local server and use a SQLite database for persistence.

    ```bash
    mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns &
    ```
    *   The `&` at the end runs the server in the background, allowing you to continue using your terminal.
    *   You can access the MLflow UI by opening `http://localhost:5000` in your web browser.
    *   To stop the server later, you might need to find its process ID (e.g., `lsof -i :5000` or `ps aux | grep mlflow`) and use `kill <PID>`.

## Usage

Once the setup is complete and the MLflow server is running, you can run each component of the pipeline sequentially. **Ensure each command is entered entirely on a single line.**

### 1. Preprocessing Data

This step resizes and prepares your images, saving them to the `data/processed` directory.

```bash
python preprocess/preprocess.py --config preprocess/config.yaml
```

### 2. Training the Model

This step trains the PyTorch CNN model, logs all metrics and parameters to MLflow, and registers the model in the MLflow Model Registry.

```bash
python training/train.py --config training/config.yaml --mlflow_config mlflow_config.yaml
```

### 3. Evaluating the Model

This step evaluates the latest trained model. If its accuracy exceeds the defined threshold (30% by default), it will set the 'prod' alias for that model version in the MLflow Model Registry and copy its artifacts to the `evaluation_area/` directory.

```bash
python evaluation/evaluate.py --config evaluation/config.yaml --mlflow_config mlflow_config.yaml
```

### 4. Running Inference

This step loads the model currently aliased as 'prod' from the MLflow Model Registry and performs inference on a single input image.

```bash
python inference/infer.py --config inference/config.yaml --mlflow_config mlflow_config.yaml --image_path data/test/ok_front/cast_ok_0_10.jpeg
```
*   **Important**: Replace `data/test/ok_front/cast_ok_0_10.jpeg` with the actual path to an image you want to use for inference.

## Configuration

Each script uses a `config.yaml` file for configurable parameters. You can modify these files to adjust behavior:

*   `preprocess/config.yaml`: Image size, normalization parameters, data directories.
*   `training/config.yaml`: Model name, batch size, epochs, learning rate.
*   `evaluation/config.yaml`: Model name, alias to set ('prod'), accuracy threshold, output directory for evaluated models.
*   `inference/config.yaml`: Model name, alias to load ('prod'), and the path for the inference image.
*   `mlflow_config.yaml`: MLflow tracking URI and artifact location.

## GitHub Actions (CI/CD)

The `.github/workflows/defect_pipeline.yml` file defines a GitHub Actions workflow that automates the entire pipeline. It is triggered on pushes to the `main` branch.

The workflow performs the following steps:
1.  Checks out the repository.
2.  Sets up Python.
3.  Installs dependencies.
4.  Starts a local MLflow Tracking Server.
5.  Runs the `preprocess.py` script.
6.  Runs the `train.py` script.
7.  Runs the `evaluate.py` script (which handles setting the 'prod' alias).
8.  Includes a placeholder for manual model promotion (as alias setting is handled by `evaluate.py`).
9.  Runs an example inference step.
10. Archives MLflow artifacts.

## Troubleshooting



```
mlflow server --host 0.0.0.0 --port 5000
```

*   **`zsh: command not found: --mlflow_config`**: This error occurs if your shell interprets the command as multiple lines. Ensure the entire command (e.g., `python training/train.py --config training/config.yaml --mlflow_config mlflow_config.yaml`) is on a single line.
*   **`TypeError: save() got an unexpected keyword argument 'tags'` or `'registered_model_tags'`**: This indicates an outdated MLflow installation. Upgrade MLflow: `pip install --upgrade mlflow`.
*   **`Max retries exceeded with url: ... (Caused by ResponseError('too many 500 error responses'))`**: This usually means the MLflow Tracking Server encountered an internal error. Stop the server (find PID with `lsof -i :5000` then `kill <PID>`) and restart it.
*   **`Registered model alias prod not found`**: This means the `evaluate.py` script hasn't yet run successfully to set the 'prod' alias, or the MLflow server was reset. Ensure you run the training and evaluation steps in order.
*   **`NameError: name 'threshold' is not defined`**: This indicates a previous error in the `evaluate.py` script. Ensure you have the latest version of the script as provided in the previous steps.
*   **Slow Evaluation/Training**:
    *   Ensure your GPU is being utilized if available. Check the output of `evaluate.py` for "Using device: cuda:0".
    *   Consider increasing the `batch_size` in `training/config.yaml` and `evaluation/config.yaml` (be mindful of GPU memory).
    *   For very large datasets, consider more advanced data loading techniques.
