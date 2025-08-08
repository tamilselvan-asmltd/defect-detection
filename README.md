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
├── train/
│   └── train.py
├── evaluation/
│   └── evaluation.py
├── inference/
│   └── inference.py
├── requirements.txt
├── README.md
└── .env
```

## Features

-   **Training (`train/train.py`):**
    -   Trains a CNN image classifier using TensorFlow.
    -   Performs training with 3 different hyperparameter sets.
    -   Tracks all runs (metrics, parameters, artifacts) in MLflow.
    -   Automatically determines and saves the model with the highest validation accuracy.

-   **Evaluation (`evaluation/evaluation.py`):**
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
    MLFLOW_TRACKING_URI=file:./mlruns
    ```
    This configures MLflow to store tracking data locally in the `mlruns` directory. For a remote MLflow server, replace `file:./mlruns` with your server's URI (e.g., `http://your-mlflow-server:5000`).

## Running Locally

Ensure your virtual environment is activated.

1.  **Run Training:**
    This will train the models, log runs to MLflow, and save the best model's information.
    ```bash
    python train/train.py
    ```

2.  **Run Evaluation:**
    This will load the best model, evaluate it, and register it in the MLflow Model Registry.
    ```bash
    python evaluation/evaluation.py
    ```

3.  **Run Inference:**
    This will pull the production model and perform sample inferences.
    ```bash
    python inference/inference.py
    ```

4.  **View MLflow UI:**
    After running the scripts, you can view the MLflow UI to inspect runs, metrics, parameters, and the model registry:
    ```bash
    mlflow ui
    ```
    Then open your web browser and navigate to `http://localhost:5000` (or the port indicated by MLflow).

## GitHub Actions CI/CD

The `.github/workflows/mlops.yml` file defines the CI/CD pipeline. It is configured to run on a `self-hosted` macOS ARM64 runner upon every push to the repository.

**Setting up a Self-Hosted Runner:**

To use the GitHub Actions workflow, you need to set up a self-hosted runner on your macOS ARM machine. Follow the official GitHub documentation for setting up self-hosted runners:

[Adding self-hosted runners - GitHub Docs](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners)

Ensure your self-hosted runner is configured with the labels `macOS` and `ARM64` (or whatever labels you define in your runner setup) to match the `runs-on` configuration in `mlops.yml`.

Once set up, any push to your repository will trigger the workflow, executing the training, evaluation, and inference steps automatically.
