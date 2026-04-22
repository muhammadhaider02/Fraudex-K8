import os
import time
import boto3
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)

app = FastAPI(title="Fraudex Inference API", version="1.0.0")

S3_BUCKET = os.getenv("S3_BUCKET", "fraudex-k8")
MODEL_RUN_ID = os.getenv("MODEL_RUN_ID", "run-1")

model = None

# ── Prometheus Metrics ─────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "fraudex_requests_total",
    "Total number of inference requests"
)
ERROR_COUNT = Counter(
    "fraudex_request_errors_total",
    "Total number of failed inference requests"
)
REQUEST_LATENCY = Histogram(
    "fraudex_request_duration_seconds",
    "Inference request latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)
FRAUD_PREDICTIONS = Counter(
    "fraudex_fraud_predictions_total",
    "Total number of transactions predicted as fraud"
)
PREDICTION_CONFIDENCE = Histogram(
    "fraudex_prediction_confidence",
    "Distribution of fraud probability scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Model performance gauges (updated via /update-metrics after each pipeline run)
FRAUD_RECALL = Gauge("fraudex_fraud_recall", "Fraud recall score")
AUC_ROC = Gauge("fraudex_auc_roc", "AUC-ROC score")
F1_SCORE = Gauge("fraudex_f1_score", "F1 score")
FALSE_POSITIVE_RATE = Gauge("fraudex_false_positive_rate", "False positive rate")

# Data drift gauges
FEATURE_DRIFT = Gauge(
    "fraudex_feature_drift_score",
    "Feature distribution drift score",
    ["feature"]
)
MISSING_VALUE_RATE = Gauge("fraudex_missing_value_rate", "Missing value rate in input data")
INPUT_ANOMALIES = Counter("fraudex_input_anomalies_total", "Total input anomalies detected")


def load_model():
    global model
    try:
        s3 = boto3.client("s3")
        key = f"models/{MODEL_RUN_ID}/xgb.joblib"
        print(f"Loading model from s3://{S3_BUCKET}/{key}")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        model = joblib.load(BytesIO(obj["Body"].read()))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None


@app.on_event("startup")
def startup_event():
    load_model()


class Transaction(BaseModel):
    TransactionAmt: float
    ProductCD: Optional[str] = None
    card4: Optional[str] = None
    card6: Optional[str] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    features: Optional[dict] = None


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    model_run_id: str
    threshold: float = 0.5


class MetricsUpdate(BaseModel):
    recall: Optional[float] = None
    auc_roc: Optional[float] = None
    f1: Optional[float] = None
    false_positive_rate: Optional[float] = None
    feature_drift: Optional[dict] = None
    missing_value_rate: Optional[float] = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "run_id": MODEL_RUN_ID}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    if model is None:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        if transaction.features:
            df = pd.DataFrame([transaction.features])
        else:
            df = pd.DataFrame([{
                "TransactionAmt": transaction.TransactionAmt,
                "TransactionAmt_log": np.log1p(transaction.TransactionAmt),
                "TransactionAmt_cent": transaction.TransactionAmt % 1,
            }])

        expected_features = model.get_booster().feature_names
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0.0
        df = df[expected_features]

        prob = float(model.predict_proba(df)[0][1])
        is_fraud = prob >= 0.5

        PREDICTION_CONFIDENCE.observe(prob)
        if is_fraud:
            FRAUD_PREDICTIONS.inc()

        REQUEST_LATENCY.observe(time.time() - start_time)

        return PredictionResponse(
            fraud_probability=prob,
            is_fraud=is_fraud,
            model_run_id=MODEL_RUN_ID,
            threshold=0.5,
        )

    except Exception as e:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update-metrics")
def update_metrics(update: MetricsUpdate):
    if update.recall is not None:
        FRAUD_RECALL.set(update.recall)
    if update.auc_roc is not None:
        AUC_ROC.set(update.auc_roc)
    if update.f1 is not None:
        F1_SCORE.set(update.f1)
    if update.false_positive_rate is not None:
        FALSE_POSITIVE_RATE.set(update.false_positive_rate)
    if update.feature_drift is not None:
        for feature, score in update.feature_drift.items():
            FEATURE_DRIFT.labels(feature=feature).set(score)
    if update.missing_value_rate is not None:
        MISSING_VALUE_RATE.set(update.missing_value_rate)
    return {"status": "metrics updated"}


@app.post("/reload-model")
def reload_model():
    load_model()
    return {"status": "reloaded", "model_loaded": model is not None, "run_id": MODEL_RUN_ID}
