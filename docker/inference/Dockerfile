FROM python:3.11.9-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    boto3 \
    joblib \
    numpy \
    pandas \
    scikit-learn \
    xgboost \
    lightgbm

COPY docker/inference/app.py ./app.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]