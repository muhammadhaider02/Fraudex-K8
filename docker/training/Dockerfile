FROM python:3.11.9

WORKDIR /app

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    xgboost \
    lightgbm \
    imbalanced-learn \
    joblib \
    boto3 \
    kfp==2.15.0

COPY pipelines/ ./pipelines/

CMD ["python3", "pipelines/pipeline.py"]