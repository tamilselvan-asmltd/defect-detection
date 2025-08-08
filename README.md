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
    A[Push to GitHub] --> B{GitHub Actions Workflow Triggered}
    B --> C[Checkout Code]
    C --> D[Setup Python Environment]
    D --> E[Install Dependencies]
    E --> F[Set MLflow Tracking URI]
    F --> G[Run Training Script (src/train.py)]
    G --> H[Log Training Metrics & Model to MLflow]
    H --> I[Run Evaluation Script (src/evaluation.py)]
    I --> J[Evaluate Model & Log Metrics to MLflow]
    J --> K{Accuracy >= Threshold?}
    K -- Yes --> L[Register Model as Production_Reg]
    K -- No --> M[Register Model as Testing_Reg]
    L --> N[Manage Production_Reg Aliases (set 'prod' to best version)]
    M --> O[Run Inference Script (src/inference.py)]
    N --> O
    O --> P[Load 'prod' Model from MLflow Registry]
    P --> Q[Perform Inference on Sample Images]
    Q --> R[End]
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

## GitHub Actions CI/CD

The `.github/workflows/mlops.yml` file defines the CI/CD pipeline. It is configured to run on a `self-hosted` macOS ARM64 runner upon every push to the repository.

**Setting up a Self-Hosted Runner:**

To use the GitHub Actions workflow, you need to set up a self-hosted runner on your macOS ARM machine. Follow the official GitHub documentation for setting up self-hosted runners:

[Adding self-hosted runners - GitHub Docs](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners)

Ensure your self-hosted runner is configured with the labels `macOS` and `ARM64` (or whatever labels you define in your runner setup) to match the `runs-on` configuration in `mlops.yml`.

Once set up, any push to your repository will trigger the workflow, executing the training, evaluation, and inference steps automatically.

    ```
    This configures MLflow to store tracking data locally in the `mlruns` directory. For a remote MLflow server, replace `file:./mlruns` with your server's URI (e.g., `http://your-mlflow-server:5000`).

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

## GitHub Actions CI/CD

The `.github/workflows/mlops.yml` file defines the CI/CD pipeline. It is configured to run on a `self-hosted` macOS ARM64 runner upon every push to the repository.

**Setting up a Self-Hosted Runner:**

To use the GitHub Actions workflow, you need to set up a self-hosted runner on your macOS ARM machine. Follow the official GitHub documentation for setting up self-hosted runners:

[Adding self-hosted runners - GitHub Docs](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners)

Ensure your self-hosted runner is configured with the labels `macOS` and `ARM64` (or whatever labels you define in your runner setup) to match the `runs-on` configuration in `mlops.yml`.

Once set up, any push to your repository will trigger the workflow, executing the training, evaluation, and inference steps automatically.
