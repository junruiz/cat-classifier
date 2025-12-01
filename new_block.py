import os
import argparse
import json
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import tensorflow as tf


def soft_ce(y_soft, logits):
    # y_soft: (B, C) teacher probabilities
    # logits: (B, C) student logits
    logp = tf.nn.log_softmax(logits, axis=-1)
    return -tf.reduce_mean(tf.reduce_sum(y_soft * logp, axis=-1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-directory", required=True)
    p.add_argument("--out-directory", required=True)
    p.add_argument("--info-file")
    args, _ = p.parse_known_args()

    d = args.data_directory
    out_dir = args.out_directory
    os.makedirs(out_dir, exist_ok=True)

    # EI provides these already split (do NOT merge train+test and re-split)
    X_tr = np.load(os.path.join(d, "X_split_train.npy")).astype(np.float32)
    Y_tr = np.load(os.path.join(d, "Y_split_train.npy"))
    X_te = np.load(os.path.join(d, "X_split_test.npy")).astype(np.float32)
    Y_te = np.load(os.path.join(d, "Y_split_test.npy"))

    y_tr = np.argmax(Y_tr, axis=1).astype(int)
    y_te = np.argmax(Y_te, axis=1).astype(int)
    num_classes = int(Y_tr.shape[1])
    input_dim = int(X_tr.shape[1])

    # Make a validation split from EI's training split only
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=42
        )
    except ValueError:
        X_train, y_train = X_tr, y_tr
        X_val, y_val = X_te, y_te  # fallback only

    # -----------------------------
    # 1) Train Teacher (RF)
    # -----------------------------
    teacher = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    teacher.fit(X_train, y_train)

    teacher_val_acc = float(teacher.score(X_val, y_val)) if X_val.size else 0.0
    teacher_test_acc = float(teacher.score(X_te, y_te)) if X_te.size else 0.0

    # Soft labels (probabilities)
    P_train = teacher.predict_proba(X_train).astype(np.float32)  # (N, C)
    P_val = teacher.predict_proba(X_val).astype(np.float32)

    # Safety: ensure columns = num_classes (sklearn sometimes omits absent classes in split)
    # We'll expand to full num_classes if needed.
    def expand_probs(P, present_classes):
        if P.shape[1] == num_classes:
            return P
        full = np.zeros((P.shape[0], num_classes), dtype=np.float32)
        for col, cls in enumerate(present_classes):
            full[:, int(cls)] = P[:, col]
        # renormalize
        s = np.sum(full, axis=1, keepdims=True)
        full = full / np.clip(s, 1e-9, None)
        return full

    present = teacher.classes_
    P_train = expand_probs(P_train, present)
    P_val = expand_probs(P_val, present)

    # Optionally save teacher for debugging (not for MCU deploy)
    with open(os.path.join(out_dir, "teacher_rf.pkl"), "wb") as f:
        pickle.dump(teacher, f)

    # -----------------------------
    # 2) Train Student (small NN) to mimic teacher probs
    # -----------------------------
    student = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes)  # logits
    ])

    student.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=soft_ce,
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")]
    )

    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        )
    ]

    student.fit(
        X_train, P_train,
        validation_data=(X_val, P_val),
        epochs=60,
        batch_size=32,
        verbose=2,
        callbacks=cb,
    )

    # Evaluate student on true labels (hard eval)
    # Convert logits -> probs
    student_probs_te = tf.nn.softmax(student(X_te, training=False), axis=-1).numpy()
    student_pred_te = np.argmax(student_probs_te, axis=1)
    student_test_acc = float(np.mean(student_pred_te == y_te)) if X_te.size else 0.0

    # -----------------------------
    # 3) Export Student to TFLite (INT8) for MCU
    # -----------------------------
    converter = tf.lite.TFLiteConverter.from_keras_model(student)

    # Representative dataset for quantization
    def rep_data():
        n = min(300, X_train.shape[0])
        for i in range(n):
            yield [X_train[i:i+1].astype(np.float32)]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_int8 = converter.convert()
    with open(os.path.join(out_dir, "model_quantized_int8_io.tflite"), "wb") as f:
        f.write(tflite_int8)

    # Also export float model (optional; useful for debugging)
    converter_fp = tf.lite.TFLiteConverter.from_keras_model(student)
    tflite_fp = converter_fp.convert()
    with open(os.path.join(out_dir, "model.tflite"), "wb") as f:
        f.write(tflite_fp)

    # Write log for EI
    with open(os.path.join(out_dir, "training_log.json"), "w") as f:
        json.dump(
            {
                "teacher_val_acc": teacher_val_acc,
                "teacher_test_acc": teacher_test_acc,
                "student_test_acc_hard": student_test_acc,
                "train_samples": int(len(y_train)),
                "val_samples": int(len(y_val)),
                "test_samples": int(len(y_te)),
                "input_dim": input_dim,
                "num_classes": num_classes,
                "outputs": ["model_quantized_int8_io.tflite", "model.tflite", "teacher_rf.pkl"],
            },
            f,
            indent=2
        )


if __name__ == "__main__":
    main()
