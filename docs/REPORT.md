# Fraudex-K8: Fraud Detection System Report

**Dataset:** IEEE CIS Fraud Detection (590,540 transactions, 434 features post-merge, 3.5% fraud rate)
**Infrastructure:** AWS EC2 g4dn.2xlarge, k3s v1.34.6, KFP 2.15.0, S3 artifact storage

---

## 1. Environment Setup

Kubeflow Pipelines v2.15.0 was deployed on k3s v1.34.6 running on an AWS EC2 g4dn.2xlarge instance with 8 vCPUs, 32GB RAM and 250GB gp3 EBS storage. The instance used the Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04, 20250701) which ships with CUDA, NVIDIA drivers and Docker pre-installed.

Minikube was initially attempted but abandoned due to persistent security context failures. KFP 2.15.0 hardcodes `runAsNonRoot: true` in every generated Argo workflow pod spec, but the argoexec init container runs as root. No configmap patch can override an explicitly set field in the pod spec. k3s was selected as a lightweight production-grade alternative, resolving the networking and image pull reliability issues encountered with Minikube.

The security context conflict was resolved by deploying Kyverno v1.12.0 as a mutating admission webhook. A `ClusterPolicy` was applied to strip `runAsNonRoot: true` from all pods in the `kubeflow` namespace before they reach the kubelet. SeaweedFS, the internal S3-compatible artifact store introduced in KFP 2.15.0, required a service patch to expose port 9000 mapped to its native port 8333. Resource limits were increased on the SeaweedFS deployment to handle 590K-row CSV uploads without connection timeouts.

Training data was stored in S3 (`s3://fraudex-k8/data/`) and accessed by pipeline pods via an IAM instance profile attached to the EC2 instance. No credentials were hardcoded. The IAM role was detected automatically by boto3 inside each component container.

Two systemd services were created to maintain persistence across SSH sessions and EC2 reboots: `kfp-portforward` keeps the Kubeflow UI accessible on port 8080, and `fraudex-inference` runs the FastAPI inference server on port 8000. Both are configured with `Restart=always` and registered with `systemctl enable`.

All pipeline runs were executed under the `Fraudex` experiment namespace. Resource limits were set per component rather than relying on cluster defaults: the training step was allocated 16GB memory and 6 CPU cores.

---

## 2. Pipeline Design

The Fraudex pipeline is a 7-step end-to-end fraud detection system built using KFP SDK v2.15.0. Each component runs in an isolated Docker container. Intermediate artifacts are stored in SeaweedFS. Final models and explainability plots are uploaded to S3 at the end of training and evaluation.

**Data Ingestion** downloads both CSVs from S3 using boto3 with IAM role authentication, merges them on `TransactionID` using a left join and writes the merged dataset as a pipeline artifact. The merged shape is 590,540 rows by 434 columns. The ingest step is configured with 2 retries and exponential backoff.

**Data Validation** asserts presence of the `isFraud` target column and `TransactionID`, verifies the dataset is non-empty and logs four scalar metrics to the KFP dashboard: total records, fraud case count, fraud rate percentage and average missing value percentage. The validation step retries once on failure.

**Preprocessing** drops columns exceeding the 50% missing value threshold, imputes numeric columns with column medians and categorical columns with the column mode, then encodes all categoricals. High-cardinality columns with more than 50 unique values are frequency-encoded. Low-cardinality columns use label encoding.

**Feature Engineering** creates two derived transaction amount features: `TransactionAmt_log` via log1p and `TransactionAmt_cent` via modulo 1 for the fractional cent component. The dataset is then split into 80/20 stratified train and test sets using `random_state=42`.

**Model Training** conditionally applies SMOTE or class weighting based on the `imbalance_strategy` parameter, then trains three models in sequence. All three are saved locally as joblib files and uploaded to `s3://fraudex-k8/models/{run_id}/`.

**Evaluation** loads all three models, computes precision, recall, F1, AUC-ROC and confusion matrix for each, selects the best model by recall, runs SHAP TreeExplainer on a 500-sample subset and uploads the summary plot to S3. The component then pushes all evaluation metrics and feature drift scores to the inference API's `/update-metrics` endpoint so Prometheus and Grafana reflect the current run's results.

The pipeline accepts nine runtime parameters: `s3_bucket`, `s3_transaction_key`, `s3_identity_key`, `run_id`, `missing_threshold`, `test_size`, `imbalance_strategy`, `cost_sensitive` and `recall_threshold`. Every run is fully reproducible and configurable from the Kubeflow UI without any code modification.

Three pipeline versions were compiled across the project:

| Version | File | Description |
|:--------|:-----|:------------|
| v1 | `v1_fraudex.yaml` | Baseline with random stratified split |
| v2 | `v2_fraudex.yaml` | Temporal split with drift injection |
| v3 | `v3_fraudex.yaml` | Full retraining with hybrid strategy decision |

---

## 3. Data Challenges

### Missing Values

The IEEE CIS dataset has significant missingness across both files. After merging, the average missing value percentage across all columns was approximately 26%. Columns exceeding the 50% threshold were dropped in the preprocessing step, which removed a substantial portion of the V-series Vesta engineered features. The remaining numeric columns were imputed with column medians and categorical columns were imputed with the column mode. Median imputation is robust to outliers compared to mean imputation and preserves the distributional shape of the data.

### High-Cardinality Categorical Features

The identity file contains several high-cardinality categorical features including device type, browser and operating system strings. Features with more than 50 unique values were frequency-encoded, replacing each category value with its relative frequency in the training distribution. This avoids the dimensionality explosion of one-hot encoding while preserving signal about rare versus common categories. Features with 50 or fewer unique values used label encoding.

### Class Imbalance

The dataset contains approximately 20,663 fraud cases out of 590,540 total transactions, a fraud rate of 3.5%. The negative-to-positive ratio is approximately 28:1. Two strategies were compared across the four baseline runs.

**SMOTE** (Synthetic Minority Oversampling Technique) generates synthetic fraud samples by interpolating between existing fraud cases in feature space. It was applied after the train/test split to prevent data leakage. The resampling brings the training set to a 1:1 class ratio before model fitting.

**Class Weight** passes the inverse class frequency as a weight to each model's built-in weighting mechanism: `scale_pos_weight` for XGBoost and LightGBM, and `class_weight` for RandomForestClassifier and LogisticRegression. No synthetic samples are generated.

---

## 4. Models

Three models were trained per run:

**XGBoost** used 300 estimators, max depth 6, learning rate 0.05, subsample 0.8, colsample_bytree 0.8 and `aucpr` as the evaluation metric. `scale_pos_weight` was set to the fraud class weight when `cost_sensitive=True`, otherwise 1. GPU training was removed after confirming the k3s pod was CPU-only with no OpenCL support.

**LightGBM** used 300 estimators, 63 leaves, learning rate 0.05, subsample 0.8, colsample_bytree 0.8 and explicit `device_type="cpu"` to prevent the GPU trainer from activating. `scale_pos_weight` was applied identically to XGBoost.

**Hybrid (RF + LR Voting)** combined a 200-tree Random Forest and a Logistic Regression with max_iter=1000 in a soft-voting ensemble. Both sub-models used `class_weight={0: 1, 1: fn_weight}`. Soft voting averages predicted probabilities, which is more calibrated than hard voting for imbalanced datasets.

The best model per run was selected by recall, consistent with the system requirement to maximize fraud case detection.

---

## 5. Cost-Sensitive Learning

Cost-sensitive learning assigns asymmetric penalties to misclassification by setting the fraud class weight to the negative-to-positive ratio (~28x) when `cost_sensitive=True`. When `False`, all class weights are set to 1.

### Results

| Run | Imbalance Strategy | Cost-Sensitive | Best Model | Recall | F1 | AUC-ROC |
|:----|:------------------:|:--------------:|:----------:|:------:|:--:|:-------:|
| 1 | SMOTE | Yes | Hybrid | **0.8689** | 0.2388 | 0.9179 |
| 2 | SMOTE | No | Hybrid | 0.5021 | 0.5969 | 0.8885 |
| 3 | Class Weight | Yes | LightGBM | 0.8403 | 0.4166 | 0.9509 |
| 4 | Class Weight | No | LightGBM | 0.4955 | 0.6435 | **0.9519** |

### Analysis

Cost-sensitive learning was the dominant factor in driving recall. Enabling it improved recall by 36.8 percentage points for SMOTE (0.8689 vs 0.5021) and by 34.5 percentage points for class weight (0.8403 vs 0.4955). The trade-off is a significant reduction in F1 and an increase in false positives, as the model aggressively classifies borderline cases as fraud.

From a business perspective, this trade-off is appropriate. A missed fraud transaction carries a direct financial loss equal to the full transaction value. A false alarm carries only the friction cost of a declined legitimate transaction. The asymmetric cost structure in financial fraud consistently favors high recall configurations. Run 1 (SMOTE + cost-sensitive) is the recommended production configuration for recall-first use cases. Run 4 (class weight, no cost-sensitive) is preferred when balanced precision-recall and AUC-ROC matter equally, achieving the highest AUC-ROC at 0.9519 and the best F1 at 0.6435.

The best model also shifted between strategies. SMOTE runs selected the Hybrid (RF + LR) as the best model by recall, while class weight runs selected LightGBM. This suggests the two strategies interact differently with ensemble versus single-model calibration.

---

## 6. CI/CD Pipeline

The CI/CD pipeline runs on GitHub Actions and covers four stages.

**Stage 1 (Continuous Integration)** triggers on every push to main and every pull request. It installs dependencies, lints `pipeline.py` with flake8 using a 120-character line limit, validates that the pipeline compiles cleanly by running it through the KFP compiler, runs pytest against the `tests/` directory and performs schema validation by asserting the presence of required columns.

**Stage 2 (Build and Packaging)** runs only on pushes to main. It authenticates to ECR using IAM user credentials stored in GitHub Secrets, builds Docker images for the training pipeline and the inference API, tags each image with both the short commit SHA and `latest`, and pushes both tags to ECR.

**Stage 3 and Stage 4 (Deployment and Intelligent Trigger)** execute in a single SSH session to avoid connection teardown issues observed when using multiple sequential SSH steps. The session pulls the latest code, submits a new KFP run via the Python client with a timestamped `run_id`, then restarts the inference service via `sudo systemctl restart fraudex-inference`. The KFP run submission logs the Run ID as evidence of automated retraining. The `workflow_dispatch` trigger accepts a `trigger_reason` input, allowing Prometheus alerting rules to invoke the pipeline with reasons such as `drift_detected` or `performance_drop`.

Both the KFP port-forward and the inference API are managed as systemd services so they persist across SSH sessions and survive EC2 reboots without manual intervention.

---

## 7. Observability and Monitoring

### System-Level Metrics

node_exporter v1.7.0 scrapes CPU, memory, disk and network metrics from the EC2 host and exposes them on port 9100. Prometheus collects these every 15 seconds. The system health Grafana dashboard visualizes CPU usage percentage, memory usage percentage, API request rate, average API latency and error rate as time-series panels.

### Model-Level Metrics

The FastAPI inference server exposes a `/metrics` endpoint that Prometheus scrapes. Custom gauges track fraud recall, AUC-ROC, F1 and false positive rate. These gauges are updated by calling the `/update-metrics` endpoint at the end of each pipeline evaluation step. Counters track total requests, total errors and total fraud predictions. A histogram tracks prediction confidence distribution across 10 buckets.

### Data-Level Metrics

Feature drift scores are computed inside the evaluate component by comparing the current test set's TransactionAmt mean against a 100-unit baseline, and computing normalized standard deviation for card1 and addr1. These scores are pushed to labeled Prometheus gauges after each run. The missing value rate of the input data is also tracked as a gauge.

### Dashboards

Three Grafana dashboards were configured using JSON definitions stored in `monitoring/dashboards/`:

The **System Health dashboard** shows CPU and memory usage as time-series with traffic light thresholds (green below 70%, yellow to 90%, red above), alongside API request rate and latency panels.

The **Model Performance dashboard** shows recall, AUC-ROC, F1 and false positive rate as gauge panels with color-coded thresholds, plus a recall trend over time panel overlaying AUC-ROC for comparison.

The **Data Drift dashboard** shows feature drift scores as a bar gauge panel with per-feature labels, TransactionAmt drift over time, missing value rate trend and input anomaly count.

### Alert Rules

Alert rules are defined in `monitoring/alert_rules.yml` across three groups:

- `FraudRecallDrop` fires when recall falls below 0.75 for more than 5 minutes
- `HighFalsePositiveRate` fires when FPR exceeds 30% for more than 5 minutes
- `LowAUCROC` fires when AUC-ROC falls below 0.85 for more than 5 minutes
- `TransactionAmtDrift` fires when the TransactionAmt drift score exceeds 0.10 for more than 10 minutes
- `HighAPILatency` fires when average request latency exceeds 2 seconds for more than 5 minutes
- `HighCPUUsage` fires when CPU usage exceeds 90% for more than 10 minutes

Alerts are visualized in Grafana and designed to trigger the GitHub Actions `workflow_dispatch` event with the appropriate `trigger_reason`, closing the loop between monitoring and automated retraining.

---

## 8. Drift Simulation

The v2 pipeline replaces the standard random stratified split with a temporal split. The dataset is sorted by `TransactionDT`, the native transaction timestamp column in the IEEE CIS data. The earliest 70% of transactions by time form the training set and the remaining 30% form the test set. This mirrors real production conditions where a model trained on historical data is evaluated against a future distribution it has never seen.

To introduce a measurable concept shift, fraud transactions in the test set have their `TransactionAmt` multiplied by 2.5. This simulates a realistic scenario: new fraud patterns targeting higher-value transactions emerge after the model was trained. The `TransactionAmt_log` feature is recomputed accordingly to maintain consistency.

### v2 Results vs v1 Baseline

| Metric | v1 Baseline | v2 Drift | Change |
|:-------|:-----------:|:--------:|:------:|
| Recall | 0.8689 | 0.8965 | +0.028 |
| AUC-ROC | 0.9179 | 0.8896 | -0.028 |
| F1 | 0.2388 | 0.1598 | -0.079 |
| FPR | 0.1961 | 0.334 | +0.138 |

The false positive rate tripling from 19.6% to 33.4% is the clearest drift signal. The model retains high recall because the 2.5x scaling makes drifted fraud transactions easier to detect by threshold. However, it simultaneously misclassifies a much larger fraction of legitimate transactions as fraud because the decision boundary shifted into a region of feature space the model was not trained on. AUC-ROC dropping from 0.9179 to 0.8896 confirms overall ranking degradation despite the surface-level recall improvement.

The feature drift score for TransactionAmt rose from approximately 0.02 in the baseline to 0.45 in the drift run, well above the 0.10 alert threshold defined in the Prometheus rules.

---

## 9. Intelligent Retraining Strategy

The v3 pipeline implements a hybrid retraining strategy combining three independent triggers evaluated at the end of every pipeline run.

**Threshold-based trigger** fires when fraud recall falls below the configured `recall_threshold` (default 0.75). This is the most direct performance signal and ensures the system responds immediately to model degradation visible in evaluation metrics.

**Drift-based trigger** fires when the TransactionAmt feature drift score exceeds the configured `drift_threshold` (default 0.10). This catches distribution shifts before they degrade recall far enough to cross the threshold trigger, providing an earlier warning.

**Periodic trigger** fires when the current run number is divisible by `max_runs_since_retrain` (default 5). This provides a safety net that guarantees retraining even when both performance and drift remain within acceptable bounds, preventing model staleness on slowly shifting distributions.

The first trigger that fires is recorded as the retraining reason. The decision is logged to the KFP run output for traceability.

### Strategy Comparison

| Strategy | Stability | Compute Cost | Responsiveness |
|:---------|:---------:|:------------:|:--------------:|
| Threshold-only | Moderate | Low | Reactive (after degradation) |
| Periodic-only | High | High | Blind to actual need |
| Drift-only | Moderate | Low | Proactive but noisy |
| Hybrid (v3) | High | Moderate | Proactive and adaptive |

The hybrid strategy avoids the main failure modes of each individual approach. Threshold-only misses gradual drift that stays above threshold. Periodic-only wastes compute when models are healthy and misses sudden degradation between intervals. Drift-only can fire on statistical noise without actual performance impact. The hybrid triggers on whichever signal is most urgent, while the periodic backstop prevents indefinite drift.

### v3 Results

The v3 retrain run used the full dataset with a standard stratified split and SMOTE with cost-sensitive learning, matching the v1 run-1 configuration. Results:

| Metric | v2 Drift | v3 Retrain | Recovery |
|:-------|:--------:|:----------:|:--------:|
| Recall | 0.8965 | 0.8689 | Stabilized |
| AUC-ROC | 0.8896 | 0.9179 | Restored |
| F1 | 0.1598 | 0.2388 | Restored |
| FPR | 0.334 | 0.1961 | Restored |

AUC-ROC, F1 and FPR all returned to v1 baseline levels, confirming that retraining on the full distribution successfully recovered the model's original performance profile.

---

## 10. Explainability

SHAP TreeExplainer was applied to the XGBoost model from each pipeline version using a 500-sample random subset of the test set. Summary plots were uploaded to `s3://fraudex-k8/artifacts/shap/` after each run.

### v1 Baseline

The baseline model relies most heavily on `ProductCD`, `C2`, `C1` and `card6`. For `ProductCD`, the top feature: low values stretch far to the right on the SHAP axis, meaning low values of this feature strongly push predictions toward fraud. High values cluster slightly to the left, meaning they weakly suppress the fraud prediction. This is consistent with specific product categories being more vulnerable to fraud.

### v2 Drift

The feature hierarchy changed completely under drift. `card6` became the most important feature and `ProductCD`, which ranked first in the baseline, fell to ninth. `V29` rose from twelfth to fourth. The model is no longer making decisions the same way it was trained to. Features that were previously low-importance gained disproportionate weight because the injected distribution shift altered the feature correlations in the test set. This is a direct indicator of concept drift: the SHAP hierarchy diverging from the baseline signals that the model requires retraining regardless of what recall numbers show on the surface.

### v3 Retrained

After retraining on the full dataset, the feature importance hierarchy was restored. `ProductCD`, `C2`, `C1` and `card6` returned to their original top positions. The distribution of dot colors, the magnitude of SHAP values and the feature rankings are virtually identical to the v1 baseline plot. Side-by-side comparison of the v1 and v3 SHAP plots shows near-identical behavior, providing strong evidence that the hybrid retraining strategy successfully recovered the model's original decision-making profile.

---

## 11. Observations

- SMOTE with cost-sensitive learning achieved the highest recall (0.8689), making Run 1 the strongest configuration for fraud detection where missing fraud is the primary risk
- Class weight without cost-sensitive learning achieved the highest AUC-ROC (0.9519) and F1 (0.6435), making Run 4 the best configuration for balanced performance
- Cost-sensitive learning was the dominant factor in driving recall. Without it, both SMOTE and class weight produced recall scores near 0.50 regardless of the imbalance strategy
- The Hybrid model (RF + LR) won on recall for SMOTE runs while LightGBM won for class weight runs, suggesting the two strategies interact differently with ensemble versus single-model architectures
- The low F1 scores on cost-sensitive runs (0.2388 for Run 1) are expected. High recall with reduced precision is the correct operating point for fraud detection given the asymmetric cost of false negatives
- FPR tripling from 0.196 to 0.334 under drift is a clearer signal of degradation than the recall change, because the injected drift made fraud cases easier to detect while simultaneously pushing legitimate transactions into the fraud decision region
- SHAP feature hierarchy divergence between v1 and v2 is a stronger early warning signal than metric thresholds alone, because it captures behavioral drift before performance fully degrades
- SeaweedFS requires a service patch to expose port 9000 and adequate resource allocation to handle large CSV artifacts without upload timeouts in KFP 2.15.0
- KFP 2.15.0 hardcodes `runAsNonRoot: true` in generated Argo workflow pods. On clusters without a permissive pod security policy, this requires a Kyverno mutating webhook to strip the restriction before pod scheduling