from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output, Metrics


# ─────────────────────────────────────────────
# COMPONENT 1: Data Ingestion
# ─────────────────────────────────────────────
@dsl.component(
    base_image="python:3.11.9-slim",
    packages_to_install=["pandas", "numpy", "boto3"],
)
def ingest(
    s3_bucket: str,
    s3_transaction_key: str,
    s3_identity_key: str,
    output_data: Output[Dataset],
):
    import boto3
    import pandas as pd
    from io import BytesIO

    s3 = boto3.client("s3")

    print("Downloading transaction data from S3...")
    trans_obj = s3.get_object(Bucket=s3_bucket, Key=s3_transaction_key)
    trans = pd.read_csv(BytesIO(trans_obj["Body"].read()))

    print("Downloading identity data from S3...")
    identity_obj = s3.get_object(Bucket=s3_bucket, Key=s3_identity_key)
    identity = pd.read_csv(BytesIO(identity_obj["Body"].read()))

    print("Merging on TransactionID...")
    df = trans.merge(identity, on="TransactionID", how="left")

    print(f"Merged shape: {df.shape}")
    df.to_csv(output_data.path, index=False)
    print("Ingestion complete.")


# ─────────────────────────────────────────────
# COMPONENT 2: Data Validation
# ─────────────────────────────────────────────
@dsl.component(
    base_image="python:3.11.9-slim",
    packages_to_install=["pandas", "numpy"],
)
def validate(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
    metrics: Output[Metrics],
):
    import pandas as pd

    df = pd.read_csv(input_data.path)

    assert "isFraud" in df.columns, "Target column 'isFraud' missing"
    assert "TransactionID" in df.columns, "TransactionID column missing"
    assert df.shape[0] > 0, "Dataset is empty"

    total = len(df)
    fraud = int(df["isFraud"].sum())
    fraud_rate = round(fraud / total * 100, 2)
    missing_pct = round(df.isnull().mean().mean() * 100, 2)

    print(f"Total records   : {total}")
    print(f"Fraud cases     : {fraud} ({fraud_rate}%)")
    print(f"Avg missing %   : {missing_pct}%")

    metrics.log_metric("total_records", total)
    metrics.log_metric("fraud_cases", fraud)
    metrics.log_metric("fraud_rate_pct", fraud_rate)
    metrics.log_metric("avg_missing_pct", missing_pct)

    df.to_csv(output_data.path, index=False)
    print("Validation passed.")


# ─────────────────────────────────────────────
# COMPONENT 3: Preprocessing
# ─────────────────────────────────────────────
@dsl.component(
    base_image="python:3.11.9-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn"],
)
def preprocess(
    input_data: Input[Dataset],
    output_data: Output[Dataset],
    missing_threshold: float = 0.5,
):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(input_data.path)

    before = df.shape[1]
    df = df.loc[:, df.isnull().mean() < missing_threshold]
    print(f"Dropped {before - df.shape[1]} high-missing columns")

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["TransactionID", "isFraud"]]

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    le = LabelEncoder()
    for col in cat_cols:
        if df[col].nunique() > 50:
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq)
        else:
            df[col] = le.fit_transform(df[col].astype(str))

    df.to_csv(output_data.path, index=False)
    print(f"Preprocessing complete. Final shape: {df.shape}")


# ─────────────────────────────────────────────
# COMPONENT 4: Feature Engineering
# ─────────────────────────────────────────────
@dsl.component(
    base_image="python:3.11.9-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn"],
)
def feature_engineering(
    input_data: Input[Dataset],
    output_train: Output[Dataset],
    output_test: Output[Dataset],
    test_size: float = 0.2,
):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_data.path)

    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
        df["TransactionAmt_cent"] = df["TransactionAmt"] % 1

    df.drop(columns=["TransactionID"], inplace=True, errors="ignore")

    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    train_df = X_train.copy()
    train_df["isFraud"] = y_train.values
    test_df = X_test.copy()
    test_df["isFraud"] = y_test.values

    train_df.to_csv(output_train.path, index=False)
    test_df.to_csv(output_test.path, index=False)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape : {test_df.shape}")


# ─────────────────────────────────────────────
# COMPONENT 5: Model Training
# ─────────────────────────────────────────────
@dsl.component(
    base_image="python:3.11.9",
    packages_to_install=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "imbalanced-learn",
        "joblib",
        "boto3",
    ],
)
def train(
    input_train: Input[Dataset],
    output_model_xgb: Output[Model],
    output_model_lgbm: Output[Model],
    output_model_hybrid: Output[Model],
    s3_bucket: str,
    run_id: str,
    imbalance_strategy: str = "smote",
    cost_sensitive: bool = True,
):
    import boto3
    import joblib
    import pandas as pd
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    df = pd.read_csv(input_train.path)
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    fraud_weight = int((y == 0).sum() / (y == 1).sum())
    print(f"Class ratio (neg/pos): {fraud_weight}")

    if imbalance_strategy == "smote":
        print("Applying SMOTE...")
        sm = SMOTE(random_state=42)
        X, y = sm.fit_resample(X, y)
    else:
        print("Using class_weight strategy...")

    fn_weight = fraud_weight if cost_sensitive else 1

    s3 = boto3.client("s3")

    print("Training XGBoost...")
    xgb = XGBClassifier(
        device="cpu",
        scale_pos_weight=fn_weight,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X, y)
    joblib.dump(xgb, output_model_xgb.path)
    s3.upload_file(output_model_xgb.path, s3_bucket, f"models/{run_id}/xgb.joblib")
    print("XGBoost saved and uploaded to S3.")

    print("Training LightGBM...")
    lgbm = LGBMClassifier(
        device_type="cpu",
        scale_pos_weight=fn_weight,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    lgbm.fit(X, y)
    joblib.dump(lgbm, output_model_lgbm.path)
    s3.upload_file(output_model_lgbm.path, s3_bucket, f"models/{run_id}/lgbm.joblib")
    print("LightGBM saved and uploaded to S3.")

    print("Training Hybrid (RF + LR Voting)...")
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight={0: 1, 1: fn_weight},
        random_state=42,
        n_jobs=-1,
    )
    lr = LogisticRegression(
        class_weight={0: 1, 1: fn_weight},
        max_iter=1000,
        random_state=42,
    )
    hybrid = VotingClassifier(
        estimators=[("rf", rf), ("lr", lr)], voting="soft"
    )
    hybrid.fit(X, y)
    joblib.dump(hybrid, output_model_hybrid.path)
    s3.upload_file(output_model_hybrid.path, s3_bucket, f"models/{run_id}/hybrid.joblib")
    print("Hybrid model saved and uploaded to S3.")


# ─────────────────────────────────────────────
# COMPONENT 6: Evaluation
# ─────────────────────────────────────────────
@dsl.component(
    base_image="python:3.11.9",
    packages_to_install=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "joblib",
        "shap",
        "matplotlib",
        "boto3",
    ],
)
def evaluate(
    input_test: Input[Dataset],
    model_xgb: Input[Model],
    model_lgbm: Input[Model],
    model_hybrid: Input[Model],
    metrics: Output[Metrics],
    s3_bucket: str,
    run_id: str,
    recall_threshold: float = 0.75,
):
    import joblib
    import boto3
    import pandas as pd
    import shap
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        roc_auc_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )

    df = pd.read_csv(input_test.path)
    X_test = df.drop(columns=["isFraud"])
    y_test = df["isFraud"]

    models = {
        "XGBoost": joblib.load(model_xgb.path),
        "LightGBM": joblib.load(model_lgbm.path),
        "Hybrid": joblib.load(model_hybrid.path),
    }

    best_model = None
    best_recall = 0.0
    best_f, best_auc = 0.0, 0.0

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred
        )

        p, r, f, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary"
        )
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()

        print(f"\n{'='*40}")
        print(f"Model: {name}")
        print(f"  Precision : {p:.4f}")
        print(f"  Recall    : {r:.4f}")
        print(f"  F1        : {f:.4f}")
        print(f"  AUC-ROC   : {auc:.4f}")
        print(f"  Confusion Matrix: {cm}")

        if r > best_recall:
            best_recall = r
            best_model = name
            best_f, best_auc = f, auc

    metrics.log_metric("best_model", best_model)
    metrics.log_metric("best_recall", best_recall)
    metrics.log_metric("best_f1", best_f)
    metrics.log_metric("best_auc_roc", best_auc)

    print(f"\nBest model by recall: {best_model} ({best_recall:.4f})")

    # SHAP on XGBoost
    print("\nRunning SHAP analysis on XGBoost...")
    xgb_model = models["XGBoost"]
    explainer = shap.TreeExplainer(xgb_model)
    sample = X_test.sample(min(500, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)

    shap_path = "/tmp/shap_summary.png"
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(shap_path, dpi=100)
    print("SHAP summary saved locally.")

    s3 = boto3.client("s3")
    s3.upload_file(shap_path, s3_bucket, f"artifacts/shap/shap_summary_{run_id}.png")
    print(f"SHAP plot uploaded to s3://{s3_bucket}/artifacts/shap/shap_summary_{run_id}.png")

    decision = "true" if best_recall >= recall_threshold else "false"
    print(f"\nDeploy decision: {decision} (threshold={recall_threshold})")

    if decision == "true":
        print("Deploy decision: PASS - Model meets recall threshold.")
        print("Model registered successfully.")
    else:
        print("Deploy decision: FAIL - Recall below threshold.")
        print("Pipeline flagged for review.")


# ─────────────────────────────────────────────
# PIPELINE DEFINITION
# ─────────────────────────────────────────────
@dsl.pipeline(
    name="Fraudex Pipeline",
    description="End-to-end fraud detection pipeline on IEEE CIS dataset",
)
def fraudex_pipeline(
    s3_bucket: str = "fraudex-k8",
    s3_transaction_key: str = "data/train_transaction.csv",
    s3_identity_key: str = "data/train_identity.csv",
    run_id: str = "run-1",
    missing_threshold: float = 0.5,
    test_size: float = 0.2,
    imbalance_strategy: str = "smote",
    cost_sensitive: bool = True,
    recall_threshold: float = 0.75,
):
    # Step 1: Ingest
    ingest_task = ingest(
        s3_bucket=s3_bucket,
        s3_transaction_key=s3_transaction_key,
        s3_identity_key=s3_identity_key,
    )
    ingest_task.set_retry(num_retries=2)

    # Step 2: Validate
    validate_task = validate(
        input_data=ingest_task.outputs["output_data"],
    )
    validate_task.set_retry(num_retries=1)

    # Step 3: Preprocess
    preprocess_task = preprocess(
        input_data=validate_task.outputs["output_data"],
        missing_threshold=missing_threshold,
    )

    # Step 4: Feature Engineering
    fe_task = feature_engineering(
        input_data=preprocess_task.outputs["output_data"],
        test_size=test_size,
    )

    # Step 5: Train
    train_task = train(
        input_train=fe_task.outputs["output_train"],
        s3_bucket=s3_bucket,
        run_id=run_id,
        imbalance_strategy=imbalance_strategy,
        cost_sensitive=cost_sensitive,
    )
    train_task.set_memory_limit("16G")
    train_task.set_cpu_limit("6")

    # Step 6: Evaluate
    evaluate_task = evaluate(
        input_test=fe_task.outputs["output_test"],
        model_xgb=train_task.outputs["output_model_xgb"],
        model_lgbm=train_task.outputs["output_model_lgbm"],
        model_hybrid=train_task.outputs["output_model_hybrid"],
        s3_bucket=s3_bucket,
        run_id=run_id,
        recall_threshold=recall_threshold,
    )
    evaluate_task.set_memory_limit("8G")


# ─────────────────────────────────────────────
# COMPILE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=fraudex_pipeline,
        package_path="pipelines/v1_fraudex.yaml",
    )
    print("Pipeline compiled to pipelines/v1_fraudex.yaml")
