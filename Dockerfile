FROM edgeimpulse/learning-block:latest

RUN pip3 install --no-cache-dir scikit-learn joblib

COPY . /app
WORKDIR /app