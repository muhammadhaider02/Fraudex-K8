<div align="center">

# Fraudex-K8

**Production-Grade Fraud Detection on Kubeflow Pipelines**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![KFP](https://img.shields.io/badge/KFP-2.15.0-orange)](https://kubeflow.org)
[![Kubernetes](https://img.shields.io/badge/k3s-1.34.6-blue)](https://k3s.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-green)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-latest-green)](https://lightgbm.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

End-to-end fraud detection system on the IEEE CIS Fraud Detection dataset. Features a 7-step Kubeflow pipeline with automated data validation, class imbalance handling, cost-sensitive learning, multi-model training, SHAP explainability, and S3 artifact persistence.

</div>

---

## Results

Trained on [IEEE CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (590,540 transactions, 434 features after merge):

| Run | Imbalance Strategy | Cost-Sensitive | Best Model | Recall | F1 | AUC-ROC |
|:----|:------------------:|:--------------:|:----------:|:------:|:--:|:-------:|
| run-1 | SMOTE | Yes | Hybrid | **0.8689** | 0.2388 | 0.9179 |
| run-2 | SMOTE | No | Hybrid | 0.5021 | 0.5969 | 0.8885 |
| run-3 | Class Weight | Yes | LightGBM | 0.8403 | 0.4166 | 0.9509 |
| run-4 | Class Weight | No | LightGBM | 0.4955 | 0.6435 | **0.9519** |

Run-1 achieved the highest recall (0.8689), the primary metric for fraud detection. Run-4 achieved the highest AUC-ROC (0.9519) with the best precision-recall balance. Cost-sensitive learning consistently improved recall at the cost of F1.

---

## Hardware

| | |
|:---|:---|
| Instance | g4dn.2xlarge |
| vCPUs | 8 |
| RAM | 32 GB |
| GPU | NVIDIA T4 (16 GB VRAM) |
| Storage | 250 GB gp3 EBS |
| AMI | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20250701 |
| Kubernetes | k3s v1.34.6 |
| KFP | 2.15.0 |
| Python | 3.11.9 |

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Data Challenges](#data-challenges)
- [Models](#models)
- [Experimental Design](#experimental-design)
- [Results](#results)
- [Setup](#setup)
- [Running Experiments](#running-experiments)
- [Artifact Storage](#artifact-storage)
- [Project Structure](#project-structure)

---

## Overview

Fraudex-K8 is a production-grade fraud detection system built on the IEEE CIS dataset. The system addresses three core challenges in fraud detection: extreme class imbalance (3.5% fraud rate), high feature dimensionality (434 features post-merge), and the asymmetric cost of false negatives versus false positives.

The pipeline runs on k3s on AWS EC2, with training data and model artifacts persisted to S3. All runs are tracked in Kubeflow's experiment dashboard with full parameter and metric logging.

---

## Pipeline Architecture

```
ingest (S3 download + merge)
    └── output_data (590,540 × 434)
            └── validate (schema + fraud rate checks)
                    ├── metrics (total_records, fraud_rate_pct, avg_missing_pct)
                    └── output_data
                            └── preprocess (missing value imputation + encoding)
                                    └── output_data
                                            └── feature-engineering (log/cent features + train/test split)
                                                    ├── output_train (80%)
                                                    └── output_test (20%)
                                                            └── train (SMOTE or class_weight + 3 models)
                                                                    ├── output_model_xgb
                                                                    ├── output_model_lgbm
                                                                    └── output_model_hybrid
                                                                            └── evaluate (metrics + SHAP)
                                                                                    └── metrics artifact
```

Each component runs in an isolated Docker container. Intermediate artifacts are stored in SeaweedFS (KFP internal store). Final models and SHAP plots are uploaded to S3.

---

## Data Challenges

**Missing Values:** Columns with more than 50% missing values are dropped. Numeric columns are imputed with median, categorical with mode.

**High-Cardinality Categoricals:** Columns with more than 50 unique values are frequency-encoded. Low-cardinality columns use label encoding.

**Class Imbalance:** The dataset has a 3.5% fraud rate (~20,700 fraud cases out of 590,540 transactions). Two strategies are compared:

- **SMOTE:** Synthetic oversampling applied to the training set before model fitting
- **Class Weight:** Inverse class frequency weights passed to each model's `scale_pos_weight` or `class_weight` parameter

**Cost-Sensitive Learning:** When `cost_sensitive=True`, the fraud class weight (`scale_pos_weight`) is set to the negative/positive ratio (~28x), penalizing false negatives heavily. When `False`, weights are set to 1.

---

## Models

Three models are trained per run:

**XGBoost:** gradient boosting with `scale_pos_weight`, 300 estimators, max depth 6, learning rate 0.05.

**LightGBM:** gradient boosting with `scale_pos_weight`, 300 estimators, 63 leaves, learning rate 0.05.

**Hybrid (RF + LR Voting):** soft voting ensemble of Random Forest (200 trees) and Logistic Regression, both with class weights. Provides interpretability alongside tree-based power.

The best model per run is selected by recall, the primary metric. All three models are saved to S3 regardless.

---

## Experimental Design

Four runs under the `Fraudex` experiment, all with `missing_threshold=0.5`, `test_size=0.2`, `recall_threshold=0.75`:

| Run | Name | imbalance_strategy | cost_sensitive |
|:----|:-----|:------------------:|:--------------:|
| 1 | fraudex-v1-smote-cost-run-1 | smote | true |
| 2 | fraudex-v1-smote-run-2 | smote | false |
| 3 | fraudex-v1-cw-cost-run-3 | class_weight | true |
| 4 | fraudex-v1-cw-run-4 | class_weight | false |

---

## Setup

### Prerequisites

- AWS account with EC2 and S3 access
- Git
- Python 3.11+
- uv

### 1. S3 Setup

Create an S3 bucket and upload the dataset:

```bash
aws s3 cp train_transaction.csv s3://fraudex-k8/data/train_transaction.csv
aws s3 cp train_identity.csv s3://fraudex-k8/data/train_identity.csv
```

### 2. EC2 Setup

Launch a g4dn.2xlarge with the Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04), attach an IAM role with `AmazonS3FullAccess`, and SSH in.

### 3. Install k3s

```bash
curl -sfL https://get.k3s.io | sh -
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown ubuntu:ubuntu ~/.kube/config
```

### 4. Deploy Kubeflow Pipelines

```bash
export PIPELINE_VERSION=2.15.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION" --server-side
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
```

Apply the Kyverno security policy to allow pipeline pods to run:

```bash
kubectl apply -f https://github.com/kyverno/kyverno/releases/download/v1.12.0/install.yaml --server-side --force-conflicts
```

Then apply the pod mutation policy to strip `runAsNonRoot`:

```bash
cat <<EOF | kubectl apply -f -
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: strip-run-as-non-root
spec:
  rules:
  - name: strip-run-as-non-root
    match:
      any:
      - resources:
          kinds:
          - Pod
          namespaces:
          - kubeflow
    mutate:
      patchStrategicMerge:
        spec:
          securityContext:
            runAsNonRoot: false
          initContainers:
          - (name): "*"
            securityContext:
              runAsNonRoot: false
EOF
```

### 5. Clone and install

```bash
git clone https://github.com/muhammadhaider02/Fraudex-K8.git
cd Fraudex-K8
pip install kfp==2.15.0
```

### 6. Access the UI

On EC2:
```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80 &
```

On your local machine:
```bash
ssh -i "Fraudex.pem" -L 8080:localhost:8080 ubuntu@<ec2-ip>
```

Open `http://localhost:8080`.

---

## Running Experiments

Submit a run from the EC2 terminal:

```bash
python3 -c "
import kfp
client = kfp.Client(host='http://localhost:8080')
run = client.create_run_from_pipeline_package(
    'pipelines/v1_fraudex.yaml',
    arguments={
        's3_bucket': 'fraudex-k8',
        's3_transaction_key': 'data/train_transaction.csv',
        's3_identity_key': 'data/train_identity.csv',
        'run_id': 'run-1',
        'imbalance_strategy': 'smote',
        'cost_sensitive': True,
    },
    run_name='fraudex-v1-smote-cost-run-1',
    experiment_name='Fraudex',
)
print('Run ID:', run.run_id)
"
```

Change `run_id`, `imbalance_strategy`, and `cost_sensitive` for each run.

---

## Artifact Storage

All artifacts persist to S3 after each run:

```
fraudex-k8/
  data/
    train_transaction.csv
    train_identity.csv
  models/
    run-1/
      xgb.joblib
      lgbm.joblib
      hybrid.joblib
    run-2/ ...
    run-3/ ...
    run-4/ ...
  artifacts/
    shap/
      shap_summary_run-1.png
      shap_summary_run-2.png
      shap_summary_run-3.png
      shap_summary_run-4.png
```

---

## Project Structure

```
Fraudex-K8/
├── pipelines/
│   ├── pipeline.py
│   └── v1_fraudex.yaml
├── docs/
│   └── REPORT.md
├── assets/
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .python-version
└── README.md
```

---

## License

For educational and research use.