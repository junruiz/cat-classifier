FROM python:3.11-slim

RUN pip install --no-cache-dir numpy scikit-learn==1.3.2

WORKDIR /app
COPY . /app

ENTRYPOINT ["python", "/app/block.py"]
