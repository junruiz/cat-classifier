import os
import argparse
import json
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def load_data(data_dir):
    X_train = np.load(os.path.join(data_dir, "X_split_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_split_test.npy"))
    Y_train = np.load(os.path.join(data_dir, "Y_split_train.npy"))
    Y_test = np.load(os.path.join(data_dir, "Y_split_test.npy"))
    y_train = Y_train[:, 0].astype(int)
    y_test = Y_test[:, 0].astype(int)
    return X_train, y_train, X_test, y_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", required=True)
    parser.add_argument("--out-directory", required=True)
    parser.add_argument("--info-file")
    parser.add_argument("--epochs")
    parser.add_argument("--learning-rate")
    args, _ = parser.parse_known_args()

    X_train, y_train, X_test, y_test = load_data(args.data_directory)

    clf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test) if X_test.size else 0.0

    os.makedirs(args.out_directory, exist_ok=True)
    with open(os.path.join(args.out_directory, "model.pkl"), "wb") as f:
        pickle.dump(clf, f)

    with open(os.path.join(args.out_directory, "training_log.json"), "w") as f:
        json.dump(
            {
                "score": float(acc),
                "train_samples": int(len(y_train)),
                "test_samples": int(len(y_test)),
            },
            f,
        )

if __name__ == "__main__":
    main()
