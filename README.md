# Defect Detection System

This project implements a defect detection system using Convolutional Neural Networks (CNNs) for image classification. It leverages MLflow for experiment tracking, model management, and deployment, providing a robust MLOps pipeline. The system includes functionalities for training, evaluation, inference, and an API for triggering model re-training.

## Table of Contents

- [Project Description and Purpose](#project-description-and-purpose)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
  - [Re-training API](#re-training-api)
- [File and Folder Structure](#file-and-folder-structure)
- [Technical Details and Technologies Used](#technical-details-and-technologies-used)
- [API Endpoints](#api-endpoints)
- [Project Logic Flow](#project-logic-flow)
- [Contribution Guidelines](#contribution-guidelines)
- [License](#license)

## Project Description and Purpose

The primary purpose of this project is to automatically identify defects in images using deep learning. It's designed to be a scalable and manageable solution for MLOps, integrating MLflow to streamline the machine learning lifecycle from experimentation to production deployment. The system can be used to:

- Train and evaluate CNN models for image classification.
- Track experiments and model performance using MLflow.
- Manage model versions and aliases in the MLflow Model Registry.
- Perform inference on new images.
- Provide an API endpoint to trigger model re-training with updated data or configurations.

## Features

- **MLflow Integration**: Comprehensive tracking of experiments, parameters, metrics, and models.
- **Model Versioning**: Manages different versions of models in the MLflow Model Registry.
- **Dynamic Model Promotion**: Automatically promotes models to "Production" or "Testing" based on evaluation thresholds.
- **Re-training API**: A Flask API endpoint to initiate model re-training on demand.
- **Configurable**: Uses `config.yaml` for easy adjustment of model parameters, training settings, and MLflow configurations.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd defect-detection
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare your data:**
    Place your image datasets in the `data/train` and `data/test` directories. The expected structure is:
    ```
    data/
    ├── train/
    │   ├── class_0/
    │   └── class_1/
    └── test/
        ├── class_0/
        └── class_1/
    ```
    Where `class_0` and `class_1` are your "OK" and "Defective" image folders, respectively.

5.  **Start MLflow Tracking Server:**
    The project expects an MLflow Tracking Server to be running. You can start one locally:
    ```bash
    mlflow ui --host 0.0.0.0 --port 5000
    ```
    The tracking UI will be accessible at `http://localhost:5000`.

## Usage

### Training

The `train.py` script trains the CNN model, logs experiments to MLflow, and saves the best model.

```bash
python src/train.py
```

This will:
- Train models with different hyperparameter sets defined in `config.yaml`.
- Log each training run to MLflow.
- Save the best performing model locally to `temp_best_model/best_cnn_model.keras`.
- Update `best_model_info.txt` with the best run's details.
- Register the best model in the MLflow Model Registry.

### Evaluation

The `evaluation.py` script evaluates the best trained model and manages its registration and aliases in the MLflow Model Registry.

```bash
python src/evaluation.py
```

This script will:
- Load the best model identified during training.
- Evaluate its performance on the test dataset.
- Register the model as `Production_Reg` or `Testing_Reg` based on the `evaluation_threshold` in `config.yaml`.
- Manage the `prod` alias for `Production_Reg`, ensuring the highest accuracy model gets this alias.

### Inference

The `inference.py` script demonstrates how to load a model from the MLflow Model Registry and perform predictions.

```bash
python src/inference.py
```

This will:
- Load the model specified by `inference_model_name` and `inference_model_alias` from `config.yaml` (e.g., `Production_Reg@prod`).
- Perform inference on example images defined in `config.yaml`.

### Re-training API

A Flask API is provided to trigger the model re-training process.

1.  **Start the Flask API:**
    ```bash
    python api.py
    ```
    The API will run on `http://0.0.0.0:5001`.

2.  **Trigger re-training:**
    You can send a POST request to the `/retrain` endpoint.
    ```bash
    curl -X POST http://localhost:5001/retrain
    ```
    This will execute the `re_train/re_train.py` script, which loads a model from MLflow, re-trains it, and potentially updates the model registry.

## File and Folder Structure

```
.
├── .github/
│   └── workflows/
│       └── mlops.yml         # GitHub Actions workflow for CI/CD
├── api.py                    # Flask API for triggering re-training
├── best_model_info.txt       # Stores run_id and path of the best trained model
├── config.yaml               # Main configuration file for the project
├── requirements.txt          # Python dependencies
├── data/                     # Placeholder for training and testing datasets
│   ├── test/
│   └── train/
├── re_train/
│   ├── re_train.py           # Script for re-training the model
│   └── re_train.yaml         # Configuration specific to re-training
└── src/
    ├── evaluation.py         # Script for model evaluation and MLflow Model Registry management
    ├── inference.py          # Script for performing model inference
    └── train.py              # Script for model training and experiment tracking
```

## Technical Details and Technologies Used

-   **Python**: The primary programming language.
-   **TensorFlow/Keras**: For building and training the Convolutional Neural Network models.
-   **MLflow**: Used extensively for:
    -   **Experiment Tracking**: Logging parameters, metrics, and artifacts.
    -   **Model Registry**: Managing model versions, stages (Production, Staging, Archived), and aliases.
-   **Flask**: A micro web framework used to create the re-training API.
-   **scikit-learn**: For calculating evaluation metrics (e.g., accuracy).
-   **Pillow**: For image preprocessing.
-   **PyYAML**: For handling configuration files.
-   **python-dotenv**: For managing environment variables (e.g., MLflow Tracking URI).
-   **GitHub Actions**: For Continuous Integration/Continuous Deployment (CI/CD) pipeline automation.

## API Endpoints

### `/retrain`

-   **Method**: `POST`
-   **Description**: Triggers the model re-training process by executing the `re_train/re_train.py` script. This is a blocking call, meaning the API will wait for the re-training to complete before returning a response.
-   **Request Body**: None
-   **Response**:
    -   `200 OK`:
        ```json
        {
            "status": "success",
            "message": "Model retraining initiated successfully.",
            "stdout": "...",
            "stderr": "..."
        }
        ```
    -   `500 Internal Server Error`:
        ```json
        {
            "status": "error",
            "message": "Error during model retraining.",
            "error_details": "...",
            "stdout": "...",
            "stderr": "..."
        }
        ```

## Project Logic Flow

The following diagram illustrates the main components and their interactions within the defect detection system:

```
```mermaid
graph TD
    subgraph Data Management
        A[Raw Image Data] --> B{Data Preprocessing};
    end

    subgraph Training & Experimentation
        B --> C[src/train.py (Model Training)];
        C --> D[MLflow Tracking Server];
        C --> E[Local Best Model (temp_best_model/best_cnn_model.keras)];
        E --> F[best_model_info.txt];
    end

    subgraph Model Evaluation & Registry
        F --> G[src/evaluation.py (Model Evaluation)];
        G --> D;
        G --> H{MLflow Model Registry};
    end

    subgraph Inference
        H -- "Production_Reg@prod" --> I[src/inference.py (Model Inference)];
        I --> J[Prediction Results];
    end

    subgraph Re-training & API
        K[Flask API (api.py)] -- POST /retrain --> L[re_train/re_train.py (Model Re-training)];
        L --> D;
        L --> H;
        H -- "Testing_Reg@re-train" --> L;
    end

    subgraph CI/CD
        M[GitHub Push] --> N[MLOps Workflow (.github/workflows/mlops.yml)];
        N --> C;
        N --> G;
        N --> I;
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#fcf,stroke:#333,stroke-width:2px
    style E fill:#fcf,stroke:#333,stroke-width:2px
    style F fill:#fcf,stroke:#333,stroke-width:2px
    style G fill:#bbf,stroke:#333,stroke-width:2px
    style H fill:#fcf,stroke:#333,stroke-width:2px
    style I fill:#bbf,stroke:#333,stroke-width:2px
    style J fill:#f9f,stroke:#333,stroke-width:2px
    style K fill:#bbf,stroke:#333,stroke-width:2px
    style L fill:#bbf,stroke:#333,stroke-width:2px
    style M fill:#f9f,stroke:#333,stroke-width:2px
    style N fill:#bbf,stroke:#333,stroke-width:2px
```


```

**Explanation of Flow:**

1.  **Data Management**: Raw image data is preprocessed and organized for training and testing.
2.  **Training & Experimentation**: `src/train.py` trains the CNN model, logging all experiment details (parameters, metrics, models) to the MLflow Tracking Server. The best model is saved locally and its details are recorded in `best_model_info.txt`.
3.  **Model Evaluation & Registry**: `src/evaluation.py` uses the best model's info to load and evaluate it. Based on performance, the model is registered in the MLflow Model Registry, potentially being promoted to `Production_Reg` with a `prod` alias.
4.  **Inference**: `src/inference.py` loads the production-ready model from the MLflow Model Registry (e.g., `Production_Reg@prod`) to make predictions on new images.
5.  **Re-training & API**: The Flask `api.py` exposes a `/retrain` endpoint. Calling this endpoint triggers `re_train/re_train.py`, which loads a `Testing_Reg@re-train` model from the registry, re-trains it, and re-evaluates, potentially updating the model in the registry.
6.  **CI/CD**: The GitHub Actions workflow (`mlops.yml`) automates the training, evaluation, and inference steps upon code pushes, ensuring continuous integration and deployment practices.

## Contribution Guidelines

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and ensure the code adheres to existing style and quality.
4.  Write appropriate tests for your changes.
5.  Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file (if present, otherwise assume standard open-source practices) for details.
```
### todo
INITIAL TRAINING
[Start] 
   ↓
[Train Model]
   ↓
[Validate During Training]
   ↓
[Evaluate on Test Set]
   ↓
[Register Model in MLflow Registry]
   ↓
[Promote to Production]

RETRAINING
[Retrain Trigger]
   ↓
[Load Base Model from Registry] or [Train from Scratch]
   ↓
[Train Model]
   ↓
[Validate During Training]
   ↓
[Compare with Production Model]
   ↓ Yes (Better?)
[Evaluate on Test Set]
   ↓ Yes (Better?)
[Register in Registry]
   ↓
[Promote to Production]

```
