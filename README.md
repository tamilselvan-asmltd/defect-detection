# MLOps Defect Detection Pipeline

This project implements an MLOps pipeline for training a Convolutional Neural Network (CNN) to detect defects in images. It leverages MLflow for experiment tracking and model registry, and GitHub Actions for Continuous Integration/Continuous Deployment (CI/CD).

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── mlops.yml
├── data/
│   ├── train/
│   │   ├── def_front/
│   │   └── ok_front/
│   └── test/
│       ├── def_front/
│       └── ok_front/
├── src/
│   ├── train.py
│   ├── evaluation.py
│   └── inference.py
├── requirements.txt
├── README.md
└── .env
```

## Features

-   **Training (`src/train.py`):**
    -   Trains a CNN image classifier using TensorFlow.
    -   Performs training with 3 different hyperparameter sets.
    -   Tracks all runs (metrics, parameters, artifacts) in MLflow, with experiment names prefixed by `train/`.
    -   Automatically determines and saves the model with the highest validation accuracy.

-   **Evaluation (`src/evaluation.py`):**
    -   Loads the best model from training.
    -   Evaluates it on an unseen test dataset (`data/test`).
    -   Registers the model in MLflow Model Registry under "Production_Reg" (if accuracy >= 60%) or "Testing_Reg" (if accuracy < 60%).
    -   Adds a tag `acc:<evaluation_accuracy>` during registration.
    -   For "Production_Reg", compares all versions' `acc` tags and sets the alias "prod" to the model version with the highest accuracy.

-   **Inference (`inference/inference.py`):**
    -   Pulls the model from MLflow Model Registry where `alias == "prod"`.
    -   Runs inference on given input images.

-   **GitHub Actions (`.github/workflows/mlops.yml`):**
    -   CI/CD workflow that runs on a macOS ARM self-hosted runner.
    -   Automates code checkout, Python environment setup, dependency installation, training, evaluation, and inference tests.
    -   Enables caching of dependencies for faster execution.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd defect-detection
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure MLflow Tracking URI:**
    Create a `.env` file in the project root with the following content:
    ```
        This configures MLflow to store tracking data locally in the `mlruns` directory. For a remote MLflow server, replace `file:./mlruns` with your server's URI (e.g., `http://your-mlflow-server:5000`).

## Workflow Diagram

```mermaid
graph TD
    A[Developer Push to GitHub] --> B{GitHub Actions Workflow Triggered}
    B --> C[Checkout Code & Setup Environment]
    C --> D[Install Dependencies]
    D --> E[Set MLflow Tracking URI]

    E --> F[Run Training Script (src/train.py)]
    F --> G[Log Training Metrics & Model to MLflow Tracking]
    G --> H[Save Best Model Info Locally]

    H --> I[Run Evaluation Script (src/evaluation.py)]
    I --> J[Load Best Model from Training]
    J --> K[Evaluate Model on Test Data]
    K --> L[Log Evaluation Metrics to MLflow Tracking]

    L --> M{Accuracy >= 60%?}
    M -- Yes --> N[Register Model to MLflow Model Registry (Production_Reg)]
    M -- No --> O[Register Model to MLflow Model Registry (Testing_Reg)]

    N --> P[Compare Production_Reg Versions by Accuracy]
    P --> Q[Set 'prod' Alias to Best Version in Production_Reg]

    Q --> R[Run Inference Script (src/inference.py)]
    O --> R

    R --> S[Pull Model with 'prod' Alias from MLflow Registry]
    S --> T[Perform Inference on Sample Images]
    T --> U[End CI/CD Workflow]
```

## Running Locally

Ensure your virtual environment is activated.

1.  **Start MLflow Tracking Server (Optional, but Recommended for UI):**
    If you want to view the MLflow UI and track experiments locally, start the MLflow UI server in a separate terminal:
    ```bash
    mlflow ui --host 0.0.0.0 --port 5000
    ```
    Then open your web browser and navigate to `http://localhost:5000`.

2.  **Run Training:**
    This will train the models, log runs to MLflow under the `train/` experiment, and save the best model's information.
    ```bash
    python src/train.py
    ```

3.  **Run Evaluation:**
    This will load the best model, evaluate it, and register it in the MLflow Model Registry. Evaluation runs will be logged under the `eval/` experiment.
    ```bash
    python src/evaluation.py
    ```

4.  **Run Inference:**
    This will pull the production model and perform sample inferences.
    ```bash
    python src/inference.py
    ```

## Model Retraining API

This project includes a Flask API (`api.py`) that allows you to trigger model retraining and configure its parameters. This is particularly useful when running on a self-hosted runner, as the API can persist in the background.

### 1. API Overview

*   **`api.py`**: A simple Flask application that exposes a `/retrain` endpoint. When a POST request is received, it executes the `re_train.py` script.
*   **`re_train/re_train.py`**: The core retraining script. It loads model parameters (epochs, batch size, learning rate) directly from `re_train/re_train.yaml`.
*   **`re_train/re_train.yaml`**: This YAML file is where you define the default parameters for the retraining process.

    ```yaml
    # Example content of re_train/re_train.yaml
    paths:
      data_path: data

    model_parameters:
      image_size: [128, 128]
      batch_size: 32
      epochs: 12
      learning_rate: 0.0001 # Configure your learning rate here

    mlflow:
      testing_model_name: Testing_Reg
      testing_model_alias: re-train
      experiment_name: Re-train
      evaluation_threshold: 0.70
      production_model_name: Production_Reg

    inference_examples:
      defective_image: test/def_front/cast_def_0_1153.jpeg
      ok_image: test/ok_front/cast_ok_0_10.jpeg
    ```

### 2. Running the API Locally

To run the Flask API on your local machine:

1.  **Install Dependencies:** Ensure all required Python packages are installed:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the API Server:** Navigate to your project's root directory and execute:
    ```bash
    python api.py
    ```
    The API will typically start on `http://0.0.0.0:5001`.

### 3. Triggering Retraining via Postman

Once the API server is running, you can trigger the retraining process using Postman:

*   **Method:** `POST`
*   **URL:** `http://localhost:5001/retrain`
*   **Headers:** `Content-Type: application/json` (optional, as no body is sent)
*   **Body:** No body data is required. The `re_train.py` script will use the parameters defined in `re_train/re_train.yaml`.

### 4. Persistent API on Self-Hosted Runner

To ensure the Flask API runs persistently in the background on your self-hosted GitHub Actions runner, the workflow uses a dedicated script:

*   **`start_api.sh`**: This script uses `nohup` to start `api.py` in the background, ensuring it continues to run even after the GitHub Actions job completes. Its output is redirected to `api.log`.

    ```bash
    #!/bin/bash
    echo "Starting Flask API in background..."
    nohup python api.py > api.log 2>&1 &
    echo "Flask API started. Check api.log for output."
    ```

*   **`.github/workflows/mlops.yml`**: The workflow now calls `start_api.sh` to initiate the API.

    ```yaml
        - name: Start Flask API in background
          run: ./start_api.sh
    ```

**To stop the persistent Flask API on your self-hosted runner:**

You will need to manually find and terminate the process.
1.  **Find the process ID (PID):**
    ```bash
    ps aux | grep api.py
    ```
    Look for a line similar to `python api.py` and note its PID.
2.  **Kill the process:**
    ```bash
    kill <PID>
    ```
    Replace `<PID>` with the actual process ID you found.

### 5. Running `re_train.py` Directly

You can also run the retraining script directly from your terminal. It will use the parameters specified in `re_train/re_train.yaml`:

```bash
python re_train/re_train.py
```

## GitHub Actions CI/CD

The `.github/workflows/mlops.yml` file defines the CI/CD pipeline. It is configured to run on a `self-hosted` macOS ARM64 runner upon every push to the repository.

**Setting up a Self-Hosted Runner:**

To use the GitHub Actions workflow, you need to set up a self-hosted runner on your macOS ARM machine. Follow the official GitHub documentation for setting up self-hosted runners:

[Adding self-hosted runners - GitHub Docs](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners)

Ensure your self-hosted runner is configured with the labels `macOS` and `ARM64` (or whatever labels you define in your runner setup) to match the `runs-on` configuration in `mlops.yml`.

Once set up, any push to your repository will trigger the workflow, executing the training, evaluation, and inference steps automatically.
