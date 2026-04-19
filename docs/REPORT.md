# Fraudex-K8: Fraud Detection System Analysis

**Dataset:** IEEE CIS Fraud Detection (590,540 transactions, 434 features post-merge, 3.5% fraud rate)
**Pipeline:** Fraudex on Kubeflow Pipelines v2.15.0
**Infrastructure:** AWS EC2 g4dn.2xlarge, k3s v1.34.6, S3 artifact storage

---

## 1. Environment Setup

Kubeflow Pipelines v2.15.0 was deployed on k3s v1.34.6 running on an AWS EC2 g4dn.2xlarge instance (8 vCPUs, 32GB RAM, 250GB gp3 EBS). The instance used the Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04, 20250701) which ships with CUDA, NVIDIA drivers, and Docker pre-installed.

Minikube was initially attempted but abandoned after persistent issues with the argoexec init container failing due to KFP 2.15.0 hardcoding `runAsNonRoot: true` in every generated Argo workflow pod. k3s was chosen as a lightweight production-grade alternative, resolving the networking and image pull reliability issues encountered with Minikube.

The security context issue was resolved by deploying Kyverno v1.12.0 as a mutating admission webhook. A `ClusterPolicy` was applied to strip `runAsNonRoot: true` from all pods in the `kubeflow` namespace before they reach the kubelet. SeaweedFS, the internal S3-compatible artifact store introduced in KFP 2.15.0, required a service patch to expose port 9000 mapped to its native port 8333, and resource limits were increased to handle the 590K-row dataset upload without i/o timeouts.

Data was stored in S3 (`s3://fraudex-k8/data/`) and accessed by pipeline pods via an IAM instance profile (`fraudex-ec2-role`) with `AmazonS3FullAccess` attached to the EC2 instance. No credentials were hardcoded. The IAM role was detected automatically by boto3 inside each component container.

All 16 Kubeflow system pods were verified running before any pipeline work began. The dashboard was accessed at `http://localhost:8080` via kubectl port-forward combined with an SSH local tunnel from the developer machine. All pipeline runs were executed under the `Fraudex` experiment namespace.

---

## 2. Pipeline Design

The Fraudex pipeline is a 7-step end-to-end fraud detection system built using KFP SDK v2.15.0. Each component runs in an isolated Docker container on the k3s cluster. Intermediate artifacts are stored in SeaweedFS and final models and explainability plots are uploaded to S3.

**Data Ingestion** downloads both CSVs from S3 using boto3 with IAM role authentication, merges them on `TransactionID` using a left join, and writes the merged dataset as a CSV artifact. The merged shape is 590,540 rows × 434 columns.

**Data Validation** asserts presence of the `isFraud` target column and `TransactionID`, verifies the dataset is non-empty, and logs four scalar metrics to the KFP dashboard: total records, fraud case count, fraud rate percentage, and average missing value percentage.

**Preprocessing** drops columns with more than 50% missing values, imputes numeric columns with median and categorical columns with mode, and encodes categoricals. High-cardinality columns (more than 50 unique values) are frequency-encoded; low-cardinality columns use label encoding.

**Feature Engineering** creates two transaction amount features (`TransactionAmt_log` via log1p and `TransactionAmt_cent` via modulo 1 for the cent component), drops `TransactionID`, and splits into 80/20 stratified train and test sets using `random_state=42`.

**Model Training** conditionally applies SMOTE or class weighting based on the `imbalance_strategy` parameter, then trains three models: XGBoost, LightGBM, and a soft-voting Hybrid of Random Forest and Logistic Regression. All three models are saved locally as joblib files and uploaded to `s3://fraudex-k8/models/{run_id}/`. The training component is allocated 16GB memory and 6 CPU cores.

**Evaluation** loads all three models, generates predictions and probabilities on the test set, and computes precision, recall, F1, AUC-ROC, and confusion matrix for each. The best model by recall is logged to KFP metrics. SHAP TreeExplainer runs on the XGBoost model using a 500-sample subset of the test set, and the summary plot is uploaded to `s3://fraudex-k8/artifacts/shap/shap_summary_{run_id}.png`. The deploy decision (pass/fail against `recall_threshold=0.75`) is logged inline.

The pipeline accepts nine runtime parameters: `s3_bucket`, `s3_transaction_key`, `s3_identity_key`, `run_id`, `missing_threshold`, `test_size`, `imbalance_strategy`, `cost_sensitive`, and `recall_threshold`. This makes every run fully reproducible and configurable without modifying any code.

Retry policies are applied to the two most failure-prone steps: ingest retries up to 2 times with exponential backoff, and validate retries once. Resource limits are set per component rather than using cluster defaults.

---

## 3. Data Challenges

### 3.1 Missing Values

The IEEE CIS dataset has significant missingness across both files. After merging, the average missing value percentage across all columns was approximately 26%. Columns exceeding the 50% threshold were dropped in the preprocessing step, which removed a substantial portion of the V-series Vesta engineered features. The remaining numeric columns were imputed with column medians, and categorical columns were imputed with the column mode. This strategy is robust to outliers compared to mean imputation and preserves the distributional shape of the data.

### 3.2 High-Cardinality Categorical Features

The identity file contains several high-cardinality categorical features including device type, browser, and operating system strings. Features with more than 50 unique values were frequency-encoded, replacing each category value with its relative frequency in the training distribution. This avoids the dimensionality explosion of one-hot encoding while preserving signal about rare versus common categories. Features with 50 or fewer unique values used label encoding.

### 3.3 Class Imbalance

The dataset contains approximately 20,663 fraud cases out of 590,540 total transactions, a fraud rate of 3.5%. The negative-to-positive ratio is approximately 28:1. Two strategies were compared across the four runs.

**SMOTE** (Synthetic Minority Oversampling Technique) generates synthetic fraud samples by interpolating between existing fraud cases in feature space. It was applied after the train/test split to avoid data leakage. The resampling brings the training set to a 1:1 class ratio before model fitting.

**Class Weight** passes the inverse class frequency as a weight to each model's built-in weighting mechanism: `scale_pos_weight` for XGBoost and LightGBM, and `class_weight` for RandomForestClassifier and LogisticRegression. No synthetic samples are generated.

---

## 4. Model Complexity

Three models were trained per run:

**XGBoost** used 300 estimators, max depth 6, learning rate 0.05, subsample 0.8, colsample_bytree 0.8, and `aucpr` as the evaluation metric. `scale_pos_weight` was set to the fraud class weight when `cost_sensitive=True`, otherwise 1.

**LightGBM** used 300 estimators, 63 leaves, learning rate 0.05, subsample 0.8, colsample_bytree 0.8, and explicit `device_type="cpu"` to prevent the GPU trainer from activating on a node without OpenCL. `scale_pos_weight` was applied identically to XGBoost.

**Hybrid (RF + LR Voting)** combined a 200-tree Random Forest and a Logistic Regression (max_iter=1000) in a soft-voting ensemble. Both sub-models used `class_weight={0: 1, 1: fn_weight}` where `fn_weight` is the fraud class weight. Soft voting averages predicted probabilities, which is more calibrated than hard voting for imbalanced datasets.

The best model per run was selected by recall, consistent with the system requirement to maintain high recall for fraud cases.

---

## 5. Experimental Design

Four pipeline runs were executed under the `Fraudex` experiment. All runs used `missing_threshold=0.5`, `test_size=0.2`, and `recall_threshold=0.75`. The two experimental variables were `imbalance_strategy` and `cost_sensitive`.

| Run | Name | imbalance_strategy | cost_sensitive |
|:----|:-----|:------------------:|:--------------:|
| 1 | fraudex-v1-smote-cost-run-1 | smote | true |
| 2 | fraudex-v1-smote-run-2 | smote | false |
| 3 | fraudex-v1-cw-cost-run-3 | class_weight | true |
| 4 | fraudex-v1-cw-run-4 | class_weight | false |

Runs were submitted sequentially via the KFP Python client and monitored through the Kubeflow dashboard.

---

## 6. Results

### 6.1 Full Results Table

| Run | Imbalance Strategy | Cost-Sensitive | Best Model | Recall | F1 | AUC-ROC |
|:----|:------------------:|:--------------:|:----------:|:------:|:--:|:-------:|
| 1 | SMOTE | Yes | Hybrid | **0.8689** | 0.2388 | 0.9179 |
| 2 | SMOTE | No | Hybrid | 0.5021 | 0.5969 | 0.8885 |
| 3 | Class Weight | Yes | LightGBM | 0.8403 | 0.4166 | 0.9509 |
| 4 | Class Weight | No | LightGBM | 0.4955 | **0.6435** | **0.9519** |

### 6.2 Imbalance Strategy Comparison

SMOTE with cost-sensitive learning (Run 1) achieved the highest recall at 0.8689, outperforming class weight with cost-sensitive learning (Run 3) at 0.8403 by 2.9 percentage points. Both strategies produced similar AUC-ROC scores when cost-sensitive learning was applied (0.9179 vs 0.9509), with class weight producing a meaningfully higher AUC.

Without cost-sensitive learning, both strategies collapsed to near-random recall (0.5021 for SMOTE, 0.4955 for class weight), confirming that the imbalance strategy alone is insufficient — the penalty asymmetry from cost-sensitive learning is what drives high recall.

The best model also shifted between strategies. SMOTE runs selected the Hybrid (RF + LR) as the best model by recall, while class weight runs selected LightGBM. This suggests the two strategies affect the calibration and decision boundaries of individual models differently.

### 6.3 Cost-Sensitive Learning Impact

The impact of cost-sensitive learning was consistent and substantial across both imbalance strategies. Enabling it improved recall by 36.8 percentage points for SMOTE (0.8689 vs 0.5021) and by 34.5 percentage points for class weight (0.8403 vs 0.4955). The trade-off is a significant reduction in F1: cost-sensitive runs produce many more false positives as the model aggressively classifies borderline cases as fraud.

From a business perspective, this trade-off is intentional. A missed fraud transaction (false negative) carries a direct financial loss equal to the transaction value. A false alarm (false positive) carries a much smaller cost: the friction of a declined legitimate transaction. The optimal operating point depends on the ratio of these costs, which in financial fraud typically favors high recall.

### 6.4 Best Model by Recall

Run 1 (SMOTE + cost-sensitive) is the recommended production configuration for fraud detection use cases where recall is the primary objective. It correctly identifies 86.89% of all fraud cases in the test set, passing the 75% recall threshold defined in the pipeline parameters.

Run 4 (class weight, no cost-sensitive) is the recommended configuration where balanced precision-recall and AUC-ROC matter equally, achieving the highest AUC-ROC at 0.9519 and the best F1 at 0.6435.

---

## 7. SHAP Explainability

SHAP TreeExplainer was applied to the XGBoost model from each run on a 500-sample random subset of the test set. Summary plots were generated and uploaded to `s3://fraudex-k8/artifacts/shap/` for each run.

The SHAP analysis answers the question: what features most strongly push the model toward predicting fraud? TransactionAmt and its log-transformed variant consistently appear among the top features, as do several of the Vesta-engineered V-series features that survived the missing value threshold. The card and address features also contribute meaningful signal, consistent with domain knowledge that fraud often involves unfamiliar card/address combinations.

SHAP values also expose the cost-sensitive effect: in runs with `cost_sensitive=True`, the SHAP value magnitudes shift, showing that the model's internal thresholds have been pushed lower to flag more borderline cases as fraud.

---

## 8. Pipeline Versioning

One pipeline version was compiled and used across all four runs. The YAML was compiled locally using KFP SDK v2.15.0 and submitted to the cluster via the Python client.

Multiple code iterations were required before the pipeline ran successfully end-to-end. Key fixes included: migrating from local file paths to S3 for data ingestion, patching seaweedfs to expose the correct port, reducing CPU limits from 8 to 6 cores to fit within available cluster resources, switching the train and evaluate base images from `python:3.11.9-slim` to `python:3.11.9` to include the libgomp shared library required by LightGBM, removing GPU device flags from XGBoost and LightGBM since the pod was CPU-only, and removing the conditional deploy step which caused a type resolution error in KFP v2's executor.

---

## 9. Observations

- SMOTE with cost-sensitive learning achieved the highest recall (0.8689), making Run 1 the strongest configuration for fraud detection where missing fraud is the primary risk.
- Class weight without cost-sensitive learning achieved the highest AUC-ROC (0.9519) and F1 (0.6435), making Run 4 the best configuration for balanced performance.
- Cost-sensitive learning was the dominant factor in driving recall. Without it, both SMOTE and class weight produced recall scores near 0.5 regardless of the imbalance strategy.
- The Hybrid model (RF + LR) won on recall for SMOTE runs while LightGBM won for class weight runs, suggesting the two strategies interact differently with ensemble versus single-model architectures.
- The low F1 scores on cost-sensitive runs (0.2388 for Run 1) are expected rather than problematic. High recall with reduced precision is the correct operating point for fraud detection given the asymmetric cost of false negatives.
- SeaweedFS is the default artifact store in KFP 2.15.0 and replaced minio. It requires a service patch to expose port 9000 and adequate resource allocation when handling large CSV artifacts.
- KFP 2.15.0 hardcodes `runAsNonRoot: true` in generated Argo workflow pods. On clusters without a permissive pod security policy, this requires a Kyverno mutating webhook to strip the restriction before pod scheduling.
- The IEEE CIS dataset's high dimensionality (434 features after merge) and missingness (~26% average) make preprocessing the most operationally sensitive step. The 50% missing threshold removed a significant portion of the V-series features while preserving the most informative ones.