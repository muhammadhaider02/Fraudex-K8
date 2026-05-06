<div align="center">

# Fraudex-K8

**Production-Grade Fraud Detection on Kubernetes**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![KFP](https://img.shields.io/badge/Kubeflow_Pipelines-2.15.0-orange)](https://kubeflow.org)
[![k3s](https://img.shields.io/badge/k3s-1.34.6-blue)](https://k3s.io)
[![Prometheus](https://img.shields.io/badge/Prometheus-2.51.0-E6522C?logo=prometheus&logoColor=white)](https://prometheus.io)
[![Grafana](https://img.shields.io/badge/Grafana-13.0-F46800?logo=grafana&logoColor=white)](https://grafana.com)
[![CI/CD](https://github.com/muhammadhaider02/Fraudex-K8/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/muhammadhaider02/Fraudex-K8/actions/workflows/ci-cd.yml)

An end-to-end MLOps system for real-time fraud detection on the IEEE CIS dataset. Covers the full production lifecycle: automated pipeline orchestration on Kubernetes, CI/CD with GitHub Actions, drift-aware retraining and live observability via Prometheus and Grafana.

</div>

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Pipeline Versions](#pipeline-versions)
- [Monitoring](#monitoring)
- [Inference API](#inference-api)
- [Project Structure](#project-structure)

---

## Key Features

- **Kubeflow Pipeline:** 7-step DAG covering ingestion, validation, preprocessing, feature engineering, training, evaluation, and retraining decision
- **Multi-model training:** XGBoost, LightGBM and a soft-voting RF+LR hybrid trained per run with full S3 artifact persistence
- **Class imbalance handling:** SMOTE and class-weight strategies with cost-sensitive learning for asymmetric fraud penalty
- **CI/CD with GitHub Actions:** lint, unit test, Docker build to ECR, and automated KFP pipeline trigger on every push to main
- **Drift simulation:** temporal train/test split with injected distribution shift to model real-world degradation
- **Intelligent retraining:** hybrid strategy combining threshold-based, drift-based and periodic triggers
- **Prometheus + Grafana:** system health, model performance and data drift dashboards with alert rules wired to CI/CD
- **SHAP explainability:** TreeExplainer runs after every evaluation; summary plots uploaded to S3

---

## Architecture

```
        ┌──────────────────────────────────────────────────────────────┐
        │                     GitHub Actions CI/CD                     │
        │   push → lint → Docker build → push ECR → SSH → trigger KFP  │
        └──────────────────────────────┬───────────────────────────────┘
                                       │
              ┌────────────────────────▼────────────────────────┐
              │               AWS EC2 g4dn.2xlarge              │
              │                                                 │
              │  ┌───────────────────────────────────────────┐  │
              │  │  k3s + KFP 2.15                           │  │
              │  │                                           │  │
              │  │  ingest                                   │  │
              │  │   └─ validate                             │  │
              │  │       └─ preproc                          │  │
              │  │           └─ FE                           │  │
              │  │               └─ train                    │  │
              │  │                   └─ evaluate ──► S3      │  │
              │  │                       └─ retrain_decision │  │
              │  └───────────────────────────────────────────┘  │
              │                                                 │
              │        ┌──────────┐       ┌────────┐            │
              │        │Prometheus│       │Grafana │            │
              │        │  :9090   │       │ :3000  │            │
              │        └──────────┘       └────────┘            │
              │                                                 │
              │            ┌──────────────────┐                 │
              │            │ Inference API    │                 │
              │            │    :8000         │                 │
              │            └──────────────────┘                 │
              └─────────────────────────────────────────────────┘
                                       │
                          ┌────────────▼──────────────┐
                          │         AWS S3            │
                          │  data/  models/  shap/    │
                          └───────────────────────────┘
```

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- AWS account with EC2 and S3 access
- Docker
- kubectl

---

## Getting Started

```bash
git clone https://github.com/muhammadhaider02/Fraudex-K8.git
cd Fraudex-K8
uv sync
```

For full setup instructions: EC2 provisioning, k3s, Kubeflow, monitoring stack, CI/CD secrets, persistent services and teardown, see [MANUAL.md](docs/MANUAL.md).

---

## Pipeline Versions

| Version | File | Description |
|:--------|:-----|:------------|
| v1 | `v1_fraudex.yaml` | Baseline: random stratified split, 4 experimental runs |
| v2 | `v2_fraudex.yaml` | Drift simulation: temporal split with injected distribution shift |
| v3 | `v3_fraudex.yaml` | Intelligent retraining: hybrid threshold + drift + periodic strategy |

Recompile after changes:

```bash
python pipelines/pipeline.py
```

Pushing to `main` automatically triggers CI/CD: lint, Docker build, ECR push and KFP pipeline submission.

---

## Monitoring

Grafana at `http://<ec2-ip>:3000`. Three dashboards under `monitoring/dashboards/`:

| Dashboard | Panels |
|:----------|:-------|
| System Health | CPU, memory, API request rate, latency, error rate |
| Model Performance | Recall, AUC-ROC, F1, FPR gauges and recall trend over time |
| Data Drift | Feature drift scores, TransactionAmt distribution shift, missing value trend |

Alert rules in `monitoring/alert_rules.yml` fire on recall drops below 0.75, drift above 0.10 and API latency above 2s, triggering automated retraining via CI/CD.

---

## Inference API

FastAPI server on port 8000.

| Endpoint | Method | Description |
|:---------|:-------|:------------|
| `/health` | GET | Service health and model load status |
| `/metrics` | GET | Prometheus scrape endpoint |
| `/predict` | POST | Single transaction fraud prediction |
| `/update-metrics` | POST | Push evaluation metrics to Prometheus after a pipeline run |
| `/reload-model` | POST | Hot-reload model from S3 without restart |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 150.0}'
```

---

## Project Structure

```
Fraudex-K8/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── assets/
├── data/
├── docker/
│   ├── inference/
│   │   ├── app.py
│   │   └── Dockerfile
│   └── training/
│       └── Dockerfile
├── docs/
│   └── REPORT.md
├── monitoring/
│   ├── prometheus.yml
│   ├── alert_rules.yml
│   └── dashboards/
│       ├── system_health.json
│       ├── model_performance.json
│       └── data_drift.json
├── pipelines/
│   ├── pipeline.py
│   ├── v1_fraudex.yaml
│   ├── v2_fraudex.yaml
│   └── v3_fraudex.yaml
├── tests/
├── MANUAL.md
├── pyproject.toml
└── README.md
```
