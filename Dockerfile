FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy \
    scikit-learn==1.3.2 \
    tensorflow==2.15.0

WORKDIR /app
COPY . /app

ENTRYPOINT ["python", "/app/new_block.py"]
