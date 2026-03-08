# OPPE-1: MLOps Pipeline for Stock 5M Prediction

## 📘 Overview
This project implements a full **MLOps pipeline** for managing an end-to-end machine learning lifecycle using modern DevOps practices.  
It automates data versioning, feature engineering, model training, experiment tracking, and deployment readiness — all orchestrated through **GitHub Actions**, **DVC**, **MLflow**, and **Google Cloud Platform (GCP)**.

The objective of this project is to create a reproducible, scalable, and cloud-integrated ML workflow capable of training and tracking multiple feature versions for a stock market prediction use case (`stock_5m` experiment).

---

## 🚀 Key Objectives
- Automate the ML lifecycle (data → training → evaluation → registry → CI/CD).
- Track experiments and models with **MLflow**.
- Store and version datasets and models using **DVC with GCS remote**.
- Implement CI/CD using **GitHub Actions**.
- Deploy and manage artifacts via **Google Cloud Storage** and **Feast Feature Store**.

---

## 🏗️ Project Architecture
```text
mlops-oppe/
│
├── data/
│   ├── v0/                  # Raw datasets (baseline version)
│   ├── v1/                  # Incremental or enriched dataset
│   └── processed/           # Feature-engineered output files
│
├── src/
│   ├── prepare_features.py  # Data preprocessing and feature generation
│   ├── train.py             # Model training and metric logging
│   ├── hpo_and_register.py  # Hyperparameter tuning and model registry logic
│   └── utils/               # Helper scripts and configs
│
├── models/                  # Trained model artifacts
│
├── features/                # Feast feature definitions
│
├── tests/
│   └── test_features.py     # Unit tests for data and features
│
├── requirements.txt
├── dvc.yaml / .dvc/         # DVC pipeline & cache configuration
├── mlruns/                  # Local MLflow experiment tracking (file-based)
├── .github/workflows/ci.yml # CI/CD pipeline
└── README.md
````

---

## ⚙️ Tools & Technologies

| Component               | Tool/Service                   | Purpose                                      |
| ----------------------- | ------------------------------ | -------------------------------------------- |
| **Experiment Tracking** | MLflow                         | Logs metrics, artifacts, and models          |
| **Data Versioning**     | DVC + GCS                      | Handles dataset & model versioning           |
| **Feature Store**       | Feast                          | Manages feature engineering lifecycle        |
| **Orchestration**       | GitHub Actions                 | Automates CI/CD workflow                     |
| **Cloud Platform**      | GCP (Storage, Service Account) | Artifact storage & service authentication    |
| **Testing**             | Pytest                         | Validates data integrity and transformations |
| **Reporting**           | CML + GitHub Job Summary       | Auto-generates run metrics reports           |

---

## 🔧 Environment Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

### 4️⃣ Set up Google Cloud Authentication

Add your GCP service account JSON key to repo secrets:

* Go to **GitHub → Settings → Secrets and Variables → Actions**
* Add a secret:
  `GCP_SA_KEY = <contents_of_service_account_key.json>`

### 5️⃣ Configure MLflow

To run MLflow UI locally:

```bash
mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5000
```

Access at: [http://localhost:5000](http://localhost:5000)

---

## 🧩 CI/CD Workflow (`.github/workflows/ci.yml`)

This workflow automates:

* Authentication with GCP
* DVC data pull
* Environment setup & dependency installation
* Feature building (`prepare_features`)
* Model training (v0, v01 versions)
* Model registration (via `hpo_and_register`)
* Feast feature application
* Metrics reporting & artifact upload

### CI/CD Summary:

* Runs on every **push to main** or **pull request**.
* Generates reports via **GitHub Actions Summary**.
* Uses **MLflow** to fetch best model version.
* Uploads artifacts (models, metrics, predictions) for traceability.

---

## 📊 Pipeline Stages

| Stage                           | Script                    | Description                                    | Input                  | Output                                        |
| ------------------------------- | ------------------------- | ---------------------------------------------- | ---------------------- | --------------------------------------------- |
| **Data Preparation**            | `src/prepare_features.py` | Cleans and merges raw data                     | `data/v0`, `data/v1`   | `data/processed/*.parquet`                    |
| **Training (v0)**               | `src/train.py`            | Trains model on v0 features                    | `features_v0.parquet`  | `models/model_v0.joblib`, `metrics_v0.json`   |
| **Training (v01)**              | `src/train.py`            | Trains model on enriched features              | `features_v01.parquet` | `models/model_v01.joblib`, `metrics_v01.json` |
| **Hyperparameter Optimization** | `src/hpo_and_register.py` | Selects and registers best model in MLflow     | `metrics_v*.json`      | `MLflow registry`                             |
| **Feature Deployment**          | `feast apply`             | Updates the Feast feature registry             | Feast config           | Updated registry                              |
| **Reporting**                   | `CML`                     | Generates GitHub summary & optional PR comment | metrics files          | Markdown summary                              |

---

## ☁️ Cloud & Storage Configuration

### GCS Bucket

Example bucket: `mlop-oppe-1`
Used as the **DVC remote** for versioned data and models.

Mount (if needed locally):

```bash
gcsfuse mlop-oppe-1 ~/gcs/mlop-oppe-1
```

Un-mount:

```bash
fusermount -u ~/gcs/mlop-oppe-1
```

---

## 🧪 Running Tests

```bash
pytest -q
```

---

## 🩵 Troubleshooting

| Issue                                         | Root Cause                                   | Solution                                           |
| --------------------------------------------- | -------------------------------------------- | -------------------------------------------------- |
| `ModuleNotFoundError: No module named 'src'`  | Pytest can’t find the module                 | Add `export PYTHONPATH=$PWD` before `pytest`       |
| `Error: Could not find version yaml`          | `yaml` not installable via pip               | Replace `yaml` with `PyYAML` in `requirements.txt` |
| `403: Resource not accessible by integration` | PR from fork has limited GitHub token access | Use guarded CML step as shown in workflow          |
| MLflow UI shows no experiments                | Wrong working directory                      | Run `mlflow ui` from the same path as `mlruns/`    |

---

## 📈 Example Outputs

**Metrics Example (`metrics_v01.json`):**

```json
{
  "accuracy": 0.5461,
  "auc": 0.5493,
  "n_train": 1294674,
  "n_test": 323669
}
```

**Artifacts Uploaded:**

```
models/
 ├── model_v0.joblib
 ├── model_v01.joblib
 └── registry_best/
data/processed/
 ├── features_v0.parquet
 └── features_v01.parquet
```

---

## ✅ Results & Observations

* Automated pipeline integrates seamlessly across **GitHub Actions → GCP → MLflow**.
* Improved reproducibility through **DVC-managed versioning**.
* Feature iteration tracked via **Feast + MLflow**.
* CI generates automated reports and artifacts for review.

---
