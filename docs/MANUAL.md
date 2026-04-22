# Setup & Operations Guide

Complete instructions for provisioning, deploying, and operating Fraudex-K8 from scratch.

---

## Table of Contents

- [Infrastructure](#infrastructure)
- [AWS Setup](#aws-setup)
- [EC2 Provisioning](#ec2-provisioning)
- [Kubernetes & Kubeflow](#kubernetes--kubeflow)
- [Monitoring Stack](#monitoring-stack)
- [Persistent Services](#persistent-services)
- [CI/CD Configuration](#cicd-configuration)
- [Running Pipelines](#running-pipelines)
- [Inference API](#inference-api)
- [Accessing UIs](#accessing-uis)
- [Teardown](#teardown)

---

## Infrastructure

| Component | Spec |
|:----------|:-----|
| EC2 instance | g4dn.2xlarge (8 vCPU, 32 GB RAM) |
| AMI | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20250701 |
| Storage | 250 GB gp3 EBS |
| Kubernetes | k3s v1.34.6 |
| KFP | 2.15.0 |
| Prometheus | 2.51.0 |
| Grafana | 13.0 |
| Python | 3.11.9 |

---

## AWS Setup

### S3 bucket

```bash
aws s3 mb s3://fraudex-k8 --region us-east-1
aws s3 cp data/train_transaction.csv s3://fraudex-k8/data/train_transaction.csv
aws s3 cp data/train_identity.csv s3://fraudex-k8/data/train_identity.csv
```

### IAM role for EC2

Create a role named `fraudex-ec2-role` with the AWS managed policy `AmazonS3FullAccess` attached. Select **EC2** as the trusted entity. Attach this role when launching the instance — pipeline pods will use it automatically via instance metadata, no credentials needed in code.

### IAM user for CLI and CI/CD

Create a user named `fraudex-cli` with `AmazonS3FullAccess` and `AmazonEC2ContainerRegistryFullAccess` attached. Generate an access key and save the CSV. This key goes into GitHub Actions secrets and your local `aws configure`.

### ECR repositories

```bash
aws ecr create-repository --repository-name fraudex-training --region us-east-1
aws ecr create-repository --repository-name fraudex-inference --region us-east-1
```

---

## EC2 Provisioning

### Launch

- Instance type: `g4dn.2xlarge`
- AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04). Search in Community AMIs and pick the most recent date
- Storage: 250 GB gp3
- IAM instance profile: `fraudex-ec2-role`
- Security group inbound rules:

| Type | Port | Source |
|:-----|:-----|:-------|
| SSH | 22 | Your IP |
| Custom TCP | 8080 | 0.0.0.0/0 |
| Custom TCP | 8000 | 0.0.0.0/0 |
| Custom TCP | 9090 | 0.0.0.0/0 |
| Custom TCP | 3000 | 0.0.0.0/0 |

### SSH

```bash
ssh -i "Fraudex.pem" ubuntu@<ec2-public-ip>
```

### System update

```bash
sudo apt-get update && sudo apt-get upgrade -y
```

---

## Kubernetes & Kubeflow

### Install kubectl

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

### Install k3s

```bash
curl -sfL https://get.k3s.io | sh -
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown ubuntu:ubuntu ~/.kube/config
kubectl get nodes
```

### Deploy Kubeflow Pipelines

```bash
export PIPELINE_VERSION=2.15.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION" --server-side
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
kubectl get pods -n kubeflow
```

Wait until all pods show `Running`. The `proxy-agent` pod will remain in `CrashLoopBackOff` — this is expected and does not affect functionality.

### Fix minio image (KFP 2.15.0)

If the `seaweedfs` service does not expose port 9000:

```bash
kubectl patch svc seaweedfs -n kubeflow --type='json' \
  -p='[{"op":"add","path":"/spec/ports/-","value":{"name":"s3-alt","port":9000,"targetPort":8333,"protocol":"TCP"}}]'

kubectl patch deployment seaweedfs -n kubeflow --type=json \
  -p='[{"op":"replace","path":"/spec/template/spec/containers/0/resources","value":{"requests":{"memory":"512Mi","cpu":"250m"},"limits":{"memory":"2Gi","cpu":"1"}}}]'
```

### Install Kyverno (pod security fix)

KFP 2.15.0 hardcodes `runAsNonRoot: true` in Argo workflow pods. Kyverno strips this at admission time.

```bash
kubectl apply -f https://github.com/kyverno/kyverno/releases/download/v1.12.0/install.yaml --server-side --force-conflicts

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
          kinds: [Pod]
          namespaces: [kubeflow]
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

### Clone repo and install KFP SDK

```bash
git clone https://github.com/muhammadhaider02/Fraudex-K8.git
cd Fraudex-K8
pip3 install kfp==2.15.0
```

---

## Monitoring Stack

### Prometheus

```bash
wget https://github.com/prometheus/prometheus/releases/download/v2.51.0/prometheus-2.51.0.linux-amd64.tar.gz
tar xvf prometheus-2.51.0.linux-amd64.tar.gz
sudo mv prometheus-2.51.0.linux-amd64 /opt/prometheus
sudo ln -s /opt/prometheus/prometheus /usr/local/bin/prometheus
sudo mkdir -p /opt/prometheus/data
sudo chown ubuntu:ubuntu /opt/prometheus/data
```

Create `/etc/systemd/system/prometheus.service`:

```ini
[Unit]
Description=Prometheus
After=network.target

[Service]
User=ubuntu
ExecStart=/usr/local/bin/prometheus \
  --config.file=/home/ubuntu/Fraudex-K8/monitoring/prometheus.yml \
  --storage.tsdb.path=/opt/prometheus/data \
  --web.listen-address=0.0.0.0:9090
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### node_exporter

```bash
wget https://github.com/prometheus/node_exporter/releases/download/v1.7.0/node_exporter-1.7.0.linux-amd64.tar.gz
tar xvf node_exporter-1.7.0.linux-amd64.tar.gz
sudo mv node_exporter-1.7.0.linux-amd64/node_exporter /usr/local/bin/
```

Create `/etc/systemd/system/node_exporter.service`:

```ini
[Unit]
Description=Node Exporter
After=network.target

[Service]
User=ubuntu
ExecStart=/usr/local/bin/node_exporter
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Grafana

```bash
sudo apt-get install -y apt-transport-https software-properties-common
wget -q -O - https://apt.grafana.com/gpg.key | sudo apt-key add -
echo "deb https://apt.grafana.com stable main" | sudo tee /etc/apt/sources.list.d/grafana.list
sudo apt-get update && sudo apt-get install -y grafana
```

### Enable all monitoring services

```bash
sudo systemctl daemon-reload
sudo systemctl enable prometheus node_exporter grafana-server
sudo systemctl start prometheus node_exporter grafana-server
```

### Grafana setup

1. Open `http://<ec2-ip>:3000`, log in with `admin/admin`
2. Go to **Connections > Data sources > Add data source > Prometheus**
3. Set URL to `http://localhost:9090`, click **Save & test**
4. Go to **Dashboards > New > Import**, paste each JSON from `monitoring/dashboards/`

---

## Persistent Services

Two systemd services keep the KFP port-forward and inference API running across SSH sessions and EC2 reboots.

### KFP port-forward

Create `/etc/systemd/system/kfp-portforward.service`:

```ini
[Unit]
Description=KFP Port Forward
After=network.target

[Service]
User=ubuntu
ExecStart=/usr/local/bin/kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
Restart=always
RestartSec=5
Environment=KUBECONFIG=/home/ubuntu/.kube/config

[Install]
WantedBy=multi-user.target
```

### Inference API

Install dependencies first:

```bash
pip3 install fastapi uvicorn boto3 joblib pandas numpy scikit-learn xgboost lightgbm prometheus-client
```

Create `/etc/systemd/system/fraudex-inference.service`:

```ini
[Unit]
Description=Fraudex Inference API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Fraudex-K8/docker/inference
ExecStart=/home/ubuntu/.local/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
Environment=PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=S3_BUCKET=fraudex-k8
Environment=MODEL_RUN_ID=run-1

[Install]
WantedBy=multi-user.target
```

### Enable both

```bash
sudo systemctl daemon-reload
sudo systemctl enable kfp-portforward fraudex-inference
sudo systemctl start kfp-portforward fraudex-inference
sudo systemctl status kfp-portforward fraudex-inference
```

---

## CI/CD Configuration

### GitHub Actions secrets

Go to your repo > **Settings > Secrets and variables > Actions** and add:

| Secret | Description |
|:-------|:------------|
| `AWS_ACCESS_KEY_ID` | `fraudex-cli` IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | `fraudex-cli` IAM user secret key |
| `AWS_REGION` | `us-east-1` |
| `ECR_REGISTRY` | `<account-id>.dkr.ecr.us-east-1.amazonaws.com` |
| `EC2_HOST` | Current EC2 public IPv4. Update this after every start |
| `EC2_SSH_KEY` | Full contents of `Fraudex.pem` including BEGIN/END lines |

The workflow triggers on every push to `main` and runs four stages: lint and validate, Docker build and ECR push, SSH into EC2 to pull code and submit a KFP run and restart the inference service.

To trigger manually with a custom reason (e.g. from a monitoring alert):

```bash
gh workflow run ci-cd.yml -f trigger_reason=drift_detected
```

---

## Running Pipelines

### v1: Baseline

```bash
python3 -c "
import kfp
client = kfp.Client(host='http://localhost:8080')
client.create_run_from_pipeline_package(
    'pipelines/v1_fraudex.yaml',
    arguments={
        's3_bucket': 'fraudex-k8',
        's3_transaction_key': 'data/train_transaction.csv',
        's3_identity_key': 'data/train_identity.csv',
        'run_id': 'run-1',
        'imbalance_strategy': 'smote',
        'cost_sensitive': True,
        'inference_api_url': 'http://<internal-ip>:8000',
    },
    run_name='fraudex-v1-smote-cost-run-1',
    experiment_name='Fraudex',
)
"
```

Change `run_id`, `imbalance_strategy`, and `cost_sensitive`.

### v2: Drift simulation

```bash
python3 -c "
import kfp
client = kfp.Client(host='http://localhost:8080')
client.create_run_from_pipeline_package(
    'pipelines/v2_fraudex.yaml',
    arguments={
        's3_bucket': 'fraudex-k8',
        's3_transaction_key': 'data/train_transaction.csv',
        's3_identity_key': 'data/train_identity.csv',
        'run_id': 'drift-run-1',
        'train_frac': 0.7,
        'imbalance_strategy': 'smote',
        'cost_sensitive': True,
        'inference_api_url': 'http://<internal-ip>:8000',
    },
    run_name='fraudex-v2-drift-run-1',
    experiment_name='Fraudex',
)
"
```

### v3: Intelligent retraining

```bash
python3 -c "
import kfp
client = kfp.Client(host='http://localhost:8080')
client.create_run_from_pipeline_package(
    'pipelines/v3_fraudex.yaml',
    arguments={
        's3_bucket': 'fraudex-k8',
        's3_transaction_key': 'data/train_transaction.csv',
        's3_identity_key': 'data/train_identity.csv',
        'run_id': 'retrain-run-1',
        'imbalance_strategy': 'smote',
        'cost_sensitive': True,
        'recall_threshold': 0.75,
        'drift_threshold': 0.10,
        'current_run_number': 1,
        'inference_api_url': 'http://<internal-ip>:8000',
    },
    run_name='fraudex-v3-retrain-run-1',
    experiment_name='Fraudex',
)
"
```

### Recompile a pipeline

```bash
python pipelines/pipeline.py
```

Output goes to `pipelines/v3_fraudex.yaml` (or v1/v2 depending on which pipeline function is compiled).

---

## Inference API

### Test prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 150.0}'
```

### Push evaluation metrics to Prometheus

```bash
curl -X POST http://localhost:8000/update-metrics \
  -H "Content-Type: application/json" \
  -d '{
    "recall": 0.86886,
    "auc_roc": 0.917873,
    "f1": 0.238779,
    "false_positive_rate": 0.1961,
    "feature_drift": {"TransactionAmt": 0.02, "card1": 0.01, "addr1": 0.03},
    "missing_value_rate": 0.26
  }'
```

### Reload model from S3

```bash
curl -X POST http://localhost:8000/reload-model
```

---

## Accessing UIs

All UIs require either direct access (Prometheus, Grafana) or an SSH tunnel (Kubeflow).

**Kubeflow Pipelines** — requires SSH tunnel since port-forward binds to localhost:

```bash
# Run on your local machine
ssh -i "Fraudex.pem" -L 8080:localhost:8080 ubuntu@<ec2-ip>
```

Then open `http://localhost:8080`.

**Prometheus:** `http://<ec2-ip>:9090`

**Grafana:** `http://<ec2-ip>:3000`

**Inference API:** `http://<ec2-ip>:8000`

---

## Restarting After EC2 Stop/Start

The EC2 public IP changes on every stop/start. After starting the instance:

1. Update `EC2_HOST` in GitHub Actions secrets
2. SSH in with the new IP
3. k3s, Prometheus, Grafana, node_exporter, kfp-portforward and fraudex-inference all start automatically via systemd. No manual steps needed

Verify everything is up:

```bash
sudo systemctl status kfp-portforward fraudex-inference prometheus node_exporter grafana-server
kubectl get pods -n kubeflow
```

---

## Teardown

Delete all AWS resources when done to avoid charges.

```bash
# Terminate EC2 instance (also deletes the attached EBS volume)
aws ec2 terminate-instances --instance-ids <instance-id>

# Delete S3 bucket (empty it first)
aws s3 rm s3://fraudex-k8 --recursive
aws s3 rb s3://fraudex-k8

# Delete ECR repositories
aws ecr delete-repository --repository-name fraudex-training --force
aws ecr delete-repository --repository-name fraudex-inference --force

# Delete IAM user
aws iam detach-user-policy --user-name fraudex-cli --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam detach-user-policy --user-name fraudex-cli --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess
aws iam delete-access-key --user-name fraudex-cli --access-key-id <key-id>
aws iam delete-user --user-name fraudex-cli

# Delete IAM role
aws iam detach-role-policy --role-name fraudex-ec2-role --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam delete-role --role-name fraudex-ec2-role
```