import os
import argparse
import json
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-directory", required=True)
    p.add_argument("--out-directory", required=True)
    p.add_argument("--info-file")
    args, _ = p.parse_known_args()

    d = args.data_directory
    X_tr = np.load(os.path.join(d, "X_split_train.npy"))
    Y_tr = np.load(os.path.join(d, "Y_split_train.npy"))
    X_te = np.load(os.path.join(d, "X_split_test.npy"))
    Y_te = np.load(os.path.join(d, "Y_split_test.npy"))

    y_tr = np.argmax(Y_tr, axis=1).astype(int)
    y_te = np.argmax(Y_te, axis=1).astype(int)

    X_all = np.concatenate([X_tr, X_te], axis=0)
    y_all = np.concatenate([y_tr, y_te], axis=0)

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
        )
    except ValueError:
        X_train, y_train = X_all, y_all
        X_val, y_val = X_te, y_te

    clf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    val_acc = clf.score(X_val, y_val) if X_val.size else 0.0

    os.makedirs(args.out_directory, exist_ok=True)
    with open(os.path.join(args.out_directory, "model.pkl"), "wb") as f:
        pickle.dump(clf, f)

    with open(os.path.join(args.out_directory, "training_log.json"), "w") as f:
        json.dump(
            {
                "score": float(val_acc),
                "train_labels": sorted(set(y_train.tolist())),
                "val_labels": sorted(set(y_val.tolist())),
                "all_labels": sorted(set(y_all.tolist())),
                "train_samples": int(len(y_train)),
                "val_samples": int(len(y_val)),
            },
            f,
        )

if __name__ == "__main__":
    main()
