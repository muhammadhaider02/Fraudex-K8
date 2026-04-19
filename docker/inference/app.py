import os
import boto3
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from io import BytesIO

app = FastAPI(title="Fraudex Inference API", version="1.0.0")

S3_BUCKET = os.getenv("S3_BUCKET", "fraudex-k8")
MODEL_RUN_ID = os.getenv("MODEL_RUN_ID", "run-1")

model = None


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


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "run_id": MODEL_RUN_ID}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

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

        return PredictionResponse(
            fraud_probability=prob,
            is_fraud=is_fraud,
            model_run_id=MODEL_RUN_ID,
            threshold=0.5,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-model")
def reload_model():
    load_model()
    return {"status": "reloaded", "model_loaded": model is not None, "run_id": MODEL_RUN_ID}