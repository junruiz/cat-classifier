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


def _find_label_list(obj, num_classes):
    if isinstance(obj, dict):
        for k in ["labels", "categories", "class_names", "classes", "label_names"]:
            v = obj.get(k, None)
            if isinstance(v, list) and len(v) == num_classes:
                return [str(x) for x in v]
        for _, v in obj.items():
            out = _find_label_list(v, num_classes)
            if out is not None:
                return out
    elif isinstance(obj, list):
        for v in obj:
            out = _find_label_list(v, num_classes)
            if out is not None:
                return out
    return None


def load_label_names(info_file, num_classes):
    if not info_file:
        return None
    try:
        with open(info_file, "r") as f:
            info = json.load(f)
        names = _find_label_list(info, num_classes)
        return names
    except Exception:
        return None


def apply_confidence_threshold(probs, min_conf, uncertain_index=None):
    probs = probs.astype(np.float32)
    pred = np.argmax(probs, axis=1).astype(int)
    best = np.max(probs, axis=1).astype(np.float32)
    accepted = best >= float(min_conf)

    if uncertain_index is not None:
        pred2 = pred.copy()
        pred2[~accepted] = int(uncertain_index)
        return pred2, accepted

    return pred, accepted


def eval_probs(probs, y_true, min_conf=None, uncertain_index=None):
    y_true = y_true.astype(int)
    pred_raw = np.argmax(probs, axis=1).astype(int)
    acc_raw = float(np.mean(pred_raw == y_true)) if probs.size else 0.0

    out = {
        "acc_raw": acc_raw,
        "coverage": 1.0,
        "acc_thresholded": acc_raw,
        "acc_on_accepted": acc_raw,
    }

    if min_conf is None:
        return out

    pred_thr, accepted = apply_confidence_threshold(
        probs, min_conf=min_conf, uncertain_index=uncertain_index
    )
    coverage = float(np.mean(accepted)) if accepted.size else 0.0
    acc_thr = float(np.mean(pred_thr == y_true)) if probs.size else 0.0

    if np.any(accepted):
        acc_on_accepted = float(np.mean(pred_raw[accepted] == y_true[accepted]))
    else:
        acc_on_accepted = 0.0

    out.update(
        {
            "coverage": coverage,
            "acc_thresholded": acc_thr,
            "acc_on_accepted": acc_on_accepted,
        }
    )
    return out


def eval_tflite(
    tflite_path,
    X,
    y,
    int8_io=False,
    min_conf=None,
    uncertain_index=None,
):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    n = int(X.shape[0])
    correct_raw = 0
    correct_thr = 0
    accepted_cnt = 0

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
        else:
            out = out.astype(np.float32)

        p = out[0]
        pred = int(np.argmax(p))
        if pred == int(y[i]):
            correct_raw += 1

        if min_conf is None:
            continue

        best = float(np.max(p))
        accepted = best >= float(min_conf)
        if accepted:
            accepted_cnt += 1

        if (not accepted) and (uncertain_index is not None):
            pred2 = int(uncertain_index)
        else:
            pred2 = pred

        if pred2 == int(y[i]):
            correct_thr += 1

    acc_raw = correct_raw / max(1, n)

    if min_conf is None:
        return {
            "acc_raw": float(acc_raw),
            "coverage": 1.0,
            "acc_thresholded": float(acc_raw),
            "acc_on_accepted": float(acc_raw),
        }

    coverage = accepted_cnt / max(1, n)
    acc_thr = correct_thr / max(1, n)

    if accepted_cnt > 0:
        acc_on_accepted = correct_raw / accepted_cnt
    else:
        acc_on_accepted = 0.0

    return {
        "acc_raw": float(acc_raw),
        "coverage": float(coverage),
        "acc_thresholded": float(acc_thr),
        "acc_on_accepted": float(acc_on_accepted),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-directory", required=True)
    p.add_argument("--out-directory", required=True)
    p.add_argument("--info-file")
    p.add_argument("--min-confidence", type=float, default=0.60)
    p.add_argument("--uncertain-label", type=str, default="uncertain")
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

    label_names = load_label_names(args.info_file, num_classes)
    if label_names is None:
        label_names = [str(i) for i in range(num_classes)]

    uncertain_index = None
    for i, name in enumerate(label_names):
        if str(name).strip().lower() == str(args.uncertain_label).strip().lower():
            uncertain_index = i
            break

    T = 2.0
    alpha = 0.7
    min_conf = float(args.min_confidence)

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

    Yhard_tr = tf.keras.utils.to_categorical(y_tr, num_classes).astype(np.float32)
    Yhard_val = tf.keras.utils.to_categorical(y_val, num_classes).astype(np.float32)

    P_tr_T = soften_probs(P_tr, T=T)
    P_val_T = soften_probs(P_val, T=T)

    Ymix_tr = np.concatenate([Yhard_tr, P_tr_T], axis=1).astype(np.float32)
    Ymix_val = np.concatenate([Yhard_val, P_val_T], axis=1).astype(np.float32)

    student_logits = tf.keras.Sequential(
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

    class HardAccuracy(tf.keras.metrics.Metric):
        def __init__(self, num_classes, name="accuracy", **kwargs):
            super().__init__(name=name, **kwargs)
            self.num_classes = num_classes
            self.acc = tf.keras.metrics.CategoricalAccuracy()

        def update_state(self, y_mix, y_pred, sample_weight=None):
            y_hard = y_mix[:, : self.num_classes]
            y_prob = tf.nn.softmax(y_pred, axis=-1)
            return self.acc.update_state(y_hard, y_prob, sample_weight=sample_weight)

        def result(self):
            return self.acc.result()

        def reset_state(self):
            self.acc.reset_state()

    student_logits.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=distill_loss,
        metrics=[HardAccuracy(num_classes, name="accuracy")],
    )

    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ]

    student_logits.fit(
        X_tr,
        Ymix_tr,
        validation_data=(X_val, Ymix_val),
        epochs=80,
        batch_size=32,
        verbose=2,
        callbacks=cb,
    )

    val_logits = student_logits(X_val, training=False)
    val_probs = tf.nn.softmax(val_logits, axis=-1).numpy().astype(np.float32)

    stats_val = eval_probs(
        val_probs,
        y_val,
        min_conf=min_conf,
        uncertain_index=uncertain_index,
    )

    inputs = tf.keras.Input(shape=(input_dim,))
    logits = student_logits(inputs)
    probs = tf.keras.layers.Softmax()(logits)
    student_probs = tf.keras.Model(inputs, probs)

    converter_fp = tf.lite.TFLiteConverter.from_keras_model(student_probs)
    tflite_fp = converter_fp.convert()
    fp_path = os.path.join(out_dir, "model.tflite")
    with open(fp_path, "wb") as f:
        f.write(tflite_fp)

    converter = tf.lite.TFLiteConverter.from_keras_model(student_probs)

    def rep_data():
        n = min(500, X_tr.shape[0])
        for i in range(n):
            yield [X_tr[i : i + 1].astype(np.float32)]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32

    tflite_int8 = converter.convert()
    int8_path = os.path.join(out_dir, "model_quantized_int8_io.tflite")
    with open(int8_path, "wb") as f:
        f.write(tflite_int8)

    stats_tflite_fp = eval_tflite(
        fp_path,
        X_val,
        y_val,
        int8_io=False,
        min_conf=min_conf,
        uncertain_index=uncertain_index,
    )
    stats_tflite_int8 = eval_tflite(
        int8_path,
        X_val,
        y_val,
        int8_io=True,
        min_conf=min_conf,
        uncertain_index=uncertain_index,
    )

    print("Output directory:", out_dir)
    print("Saved:", fp_path)
    print("Saved:", int8_path)
    print("Saved:", os.path.join(out_dir, "teacher_rf.pkl"))
    print("Labels:", label_names)
    print("Uncertain index:", uncertain_index)
    print("Min confidence:", float(min_conf))

    print("Val acc (raw):", float(stats_val["acc_raw"]))
    print("Val acc (thresholded):", float(stats_val["acc_thresholded"]))
    print("Val coverage:", float(stats_val["coverage"]))
    print("Val acc on accepted:", float(stats_val["acc_on_accepted"]))

    print("TFLite float32 acc (raw):", float(stats_tflite_fp["acc_raw"]))
    print("TFLite float32 acc (thresholded):", float(stats_tflite_fp["acc_thresholded"]))
    print("TFLite int8 acc (raw):", float(stats_tflite_int8["acc_raw"]))
    print("TFLite int8 acc (thresholded):", float(stats_tflite_int8["acc_thresholded"]))

    with open(os.path.join(out_dir, "training_log.json"), "w") as f:
        json.dump(
            {
                "score": float(stats_tflite_fp["acc_raw"]),
                "teacher_train_acc": teacher_train_acc,
                "teacher_val_acc": teacher_val_acc,
                "val_acc_raw": float(stats_val["acc_raw"]),
                "val_acc_thresholded": float(stats_val["acc_thresholded"]),
                "val_coverage": float(stats_val["coverage"]),
                "val_acc_on_accepted": float(stats_val["acc_on_accepted"]),
                "tflite_float32_acc_raw": float(stats_tflite_fp["acc_raw"]),
                "tflite_float32_acc_thresholded": float(stats_tflite_fp["acc_thresholded"]),
                "tflite_float32_coverage": float(stats_tflite_fp["coverage"]),
                "tflite_int8_acc_raw": float(stats_tflite_int8["acc_raw"]),
                "tflite_int8_acc_thresholded": float(stats_tflite_int8["acc_thresholded"]),
                "tflite_int8_coverage": float(stats_tflite_int8["coverage"]),
                "distill_T": T,
                "distill_alpha": alpha,
                "min_confidence": float(min_conf),
                "uncertain_label": str(args.uncertain_label),
                "uncertain_index": uncertain_index,
                "train_samples": int(len(y_tr)),
                "val_samples": int(len(y_val)),
                "input_dim": input_dim,
                "num_classes": num_classes,
                "labels": label_names,
                "outputs": [
                    "model.tflite",
                    "model_quantized_int8_io.tflite",
                    "teacher_rf.pkl",
                    "training_log.json",
                ],
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
