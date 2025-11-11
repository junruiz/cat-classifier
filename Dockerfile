FROM python:3.11-slim

RUN pip install --no-cache-dir scikit-learn joblib numpy

WORKDIR /app
COPY . /app
