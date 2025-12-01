import os
import argparse
import json
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf


def soften_probs(P, T=2.0, eps=1e-6):
    P = np.clip(P, eps, 1.0)
    logP = np.log(P) / T
    P2 = np.exp(logP)
    return P2 / np.sum(P2, axis=1, keepdims=True)


def expand_probs(P, present_classes, num_classes):
    if P.shape[1] == num_classes:
        return P.astype(np.float32)
    full = np.zeros((P.shape[0], num_classes), dtype=np.float32)
    for col, cls in enumerate(present_classes):
        full[:, int(cls)] = P[:, col]
    s = np.sum(full, axis=1, keepdims=True)
    return full / np.clip(s, 1e-9, None)


def eval_tflite(tflite_path, X, y, int8_io=False):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    correct = 0
    n = int(X.shape[0])

    in_scale, in_zp = in_detail.get("quantization", (0.0, 0))
    out_scale, out_zp = out_detail.get("quantization", (0.0, 0))

    for i in range(n):
        x = X[i : i + 1].astype(np.float32)

        if int8_io:
            if in_scale and in_scale > 0:
                xq = np.round(x / in_scale + in_zp).astype(np.int32)
                xq = np.clip(xq, -128, 127).astype(np.int8)
                interpreter.set_tensor(in_detail["index"], xq)
            else:
                interpreter.set_tensor(in_detail["index"], x.astype(in_detail["dtype"]))
        else:
            interpreter.set_tensor(in_detail["index"], x.astype(in_detail["dtype"]))

        interpreter.invoke()
        out = interpreter.get_tensor(out_detail["index"])

        if out_detail["dtype"] == np.int8 and out_scale and out_scale > 0:
            out = (out.astype(np.float32) - out_zp) * out_scale

        pred = int(np.argmax(out, axis=1)[0])
        if pred == int(y[i]):
            correct += 1

    return correct / max(1, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-directory", required=True)
    p.add_argument("--out-directory", required=True)
    p.add_argument("--info-file")
    args, _ = p.parse_known_args()

    d = args.data_directory
    out_dir = args.out_directory
    os.makedirs(out_dir, exist_ok=True)

    X_tr = np.load(os.path.join(d, "X_split_train.npy")).astype(np.float32)
    Y_tr = np.load(os.path.join(d, "Y_split_train.npy"))
    X_val = np.load(os.path.join(d, "X_split_test.npy")).astype(np.float32)
    Y_val = np.load(os.path.join(d, "Y_split_test.npy"))

    y_tr = np.argmax(Y_tr, axis=1).astype(int)
    y_val = np.argmax(Y_val, axis=1).astype(int)

    num_classes = int(Y_tr.shape[1])
    input_dim = int(X_tr.shape[1])

    T = 2.0
    alpha = 0.7

    # Teacher
    teacher = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    teacher.fit(X_tr, y_tr)

    teacher_train_acc = float(teacher.score(X_tr, y_tr)) if X_tr.size else 0.0
    teacher_val_acc = float(teacher.score(X_val, y_val)) if X_val.size else 0.0

    P_tr = teacher.predict_proba(X_tr).astype(np.float32)
    P_val = teacher.predict_proba(X_val).astype(np.float32)

    present = teacher.classes_
    P_tr = expand_probs(P_tr, present, num_classes)
    P_val = expand_probs(P_val, present, num_classes)

    with open(os.path.join(out_dir, "teacher_rf.pkl"), "wb") as f:
        pickle.dump(teacher, f)

    # Student (hard + soft)
    Yhard_tr = tf.keras.utils.to_categorical(y_tr, num_classes).astype(np.float32)
    Yhard_val = tf.keras.utils.to_categorical(y_val, num_classes).astype(np.float32)

    P_tr_T = soften_probs(P_tr, T=T)
    P_val_T = soften_probs(P_val, T=T)

    Ymix_tr = np.concatenate([Yhard_tr, P_tr_T], axis=1).astype(np.float32)
    Ymix_val = np.concatenate([Yhard_val, P_val_T], axis=1).astype(np.float32)

    student = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(num_classes),
        ]
    )

    ce_hard = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction="none")

    def distill_loss(y_mix, logits):
        y_hard = y_mix[:, :num_classes]
        p_soft = y_mix[:, num_classes:]

        hard = ce_hard(y_hard, logits)
        logpT = tf.nn.log_softmax(logits / T, axis=-1)
        soft = -tf.reduce_sum(p_soft * logpT, axis=-1)

        return tf.reduce_mean((1.0 - alpha) * hard + alpha * (T * T) * soft)

    def hard_acc(y_mix, logits):
        y_hard = y_mix[:, :num_classes]
        pred = tf.nn.softmax(logits, axis=-1)
        return tf.keras.metrics.categorical_accuracy(y_hard, pred)

    student.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=distill_loss,
        metrics=[hard_acc],
    )

    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ]

    student.fit(
        X_tr,
        Ymix_tr,
        validation_data=(X_val, Ymix_val),
        epochs=80,
        batch_size=32,
        verbose=2,
        callbacks=cb,
    )

    student_probs_val = tf.nn.softmax(student(X_val, training=False), axis=-1).numpy()
    student_pred_val = np.argmax(student_probs_val, axis=1)
    student_val_acc = float(np.mean(student_pred_val == y_val)) if X_val.size else 0.0

    # Export
    converter = tf.lite.TFLiteConverter.from_keras_model(student)

    def rep_data():
        n = min(500, X_tr.shape[0])
        for i in range(n):
            yield [X_tr[i : i + 1].astype(np.float32)]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_int8 = converter.convert()
    int8_path = os.path.join(out_dir, "model_quantized_int8_io.tflite")
    with open(int8_path, "wb") as f:
        f.write(tflite_int8)

    converter_fp = tf.lite.TFLiteConverter.from_keras_model(student)
    tflite_fp = converter_fp.convert()
    fp_path = os.path.join(out_dir, "model.tflite")
    with open(fp_path, "wb") as f:
        f.write(tflite_fp)

    tflite_float32_acc = eval_tflite(fp_path, X_val, y_val, int8_io=False)
    tflite_int8_acc = eval_tflite(int8_path, X_val, y_val, int8_io=True)

    print("TFLite float32 accuracy:", float(tflite_float32_acc))
    print("TFLite int8 accuracy:", float(tflite_int8_acc))

    with open(os.path.join(out_dir, "training_log.json"), "w") as f:
        json.dump(
            {
                "teacher_train_acc": teacher_train_acc,
                "teacher_val_acc": teacher_val_acc,
                "student_val_acc_hard": student_val_acc,
                "tflite_float32_acc": float(tflite_float32_acc),
                "tflite_int8_acc": float(tflite_int8_acc),
                "distill_T": T,
                "distill_alpha": alpha,
                "train_samples": int(len(y_tr)),
                "val_samples": int(len(y_val)),
                "input_dim": input_dim,
                "num_classes": num_classes,
                "outputs": [
                    "model_quantized_int8_io.tflite",
                    "model.tflite",
                    "teacher_rf.pkl",
                ],
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
