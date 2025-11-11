import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train(X, Y, classes, **kwargs):
    clf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X, Y)
    os.makedirs("output", exist_ok=True)
    joblib.dump({"model": clf, "classes": classes}, "output/model.joblib")
    acc = clf.score(X, Y)
    return {
        "score": float(acc),
        "metrics": {"train_accuracy": float(acc)},
    }

def classify(model, X):
    clf = model["model"]
    classes = model["classes"]
    if X.ndim == 1:
        X = X.reshape(1, -1)
    probs = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    return classes[pred_idx], probs.tolist()
