import itertools
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

from data_utils import class_weight_to_sample_weight
from model_utils import build_multiclass_mlp, make_callbacks, make_optimizer


def boolish(x):
    s = str(x).strip().lower()
    return s in ("true", "1", "yes", "y", "t")


def key_from_row(row):
    return (
        round(float(row["lr"]), 12),
        round(float(row["label_smoothing"]), 12),
        round(float(row["l2_reg"]), 12),
        round(float(row["w2"]), 12),
        round(float(row["w3"]), 12),
        bool(row["use_bn"]),
    )


def parse_search_lists(args):
    lrs = [float(s) for s in args.lr_list.split(",") if s.strip()]
    smooths = [float(s) for s in args.smooth_list.split(",") if s.strip()]
    l2s = [float(s) for s in args.l2_list.split(",") if s.strip()]
    w2s = [float(s) for s in args.w2_list.split(",") if s.strip()]
    w3s = [float(s) for s in args.w3_list.split(",") if s.strip()]
    bns = [boolish(s.strip()) for s in args.bn_list.split(",") if s.strip()]

    return lrs, smooths, l2s, w2s, w3s, bns


def run_combo(
    x_train,
    x_val,
    y_train,
    y_val,
    lr,
    label_smoothing,
    l2_reg,
    w1,
    w2,
    w3,
    use_bn,
    batch_size=128,
    epochs=120,
):
    tf.keras.backend.clear_session()

    arch = "WIDE-512-256"
    act = "leakyrelu"
    init_fixed = "variancescaling"
    dropout_for_model = (0.25, 0.25)
    num_classes = 4

    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)

    model = build_multiclass_mlp(
        input_dim=x_train.shape[1],
        arch=arch,
        act=act,
        dropout=dropout_for_model,
        l2_reg=float(l2_reg),
        init_name=init_fixed,
        num_classes=num_classes,
        use_bn=bool(use_bn),
    )

    optimizer = make_optimizer(lr=float(lr))
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=float(label_smoothing))
    model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])

    cw = {0: 1.0, 1: float(w1), 2: float(w2), 3: float(w3)}
    sw_train = class_weight_to_sample_weight(y_train, cw)

    checkpoint_path = "tmp_clasico_best.keras"
    callbacks = make_callbacks(checkpoint_path)

    model.fit(
        x_train,
        y_train_oh,
        validation_data=(x_val, y_val_oh),
        epochs=int(epochs),
        batch_size=int(batch_size),
        callbacks=callbacks,
        verbose=0,
        sample_weight=sw_train,
    )

    try:
        model = tf.keras.models.load_model(checkpoint_path, compile=False)
    except Exception:
        pass

    y_val_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)
    val_macro_f1 = f1_score(y_val, y_val_pred, average="macro")

    return {
        "arch": arch,
        "act": act,
        "optimizer": "adam",
        "init": init_fixed,
        "batch": int(batch_size),
        "label_smoothing": float(label_smoothing),
        "dropout": list(dropout_for_model),
        "l2_reg": float(l2_reg),
        "lr": float(lr),
        "use_bn": bool(use_bn),
        "w0": 1.0,
        "w1": float(w1),
        "w2": float(w2),
        "w3": float(w3),
        "val_macroF1": float(val_macro_f1),
    }


def load_done_combinations(results_csv, resume):
    done = set()

    if resume and os.path.exists(results_csv):
        previous = pd.read_csv(results_csv)
        for _, row in previous.iterrows():
            try:
                done.add(key_from_row(row))
            except Exception:
                pass
        print(f"[RESUME] Encontradas {len(done)} combinaciones ya guardadas en {results_csv}. Reanudando…\n")

    elif resume:
        print(f"[RESUME] --resume activado, pero no existe {results_csv}. Iniciando desde cero.\n")

    else:
        if os.path.exists(results_csv):
            os.remove(results_csv)
        print("[RESUME] Modo normal: se reinicia el sweep.\n")

    return done


def run_sweep(args, x_train, x_val, y_train, y_val):
    lrs, smooths, l2s, w2s, w3s, bns = parse_search_lists(args)
    expected_total = len(lrs) * len(smooths) * len(l2s) * len(w2s) * len(w3s) * len(bns)

    print(f"\n[GRID] Combinaciones esperadas: {expected_total}\n")

    done = load_done_combinations(args.results_csv, args.resume)

    executed = 0
    skipped = 0

    for lr, sm, l2_reg, w2, w3, use_bn in itertools.product(lrs, smooths, l2s, w2s, w3s, bns):
        key = (
            round(float(lr), 12),
            round(float(sm), 12),
            round(float(l2_reg), 12),
            round(float(w2), 12),
            round(float(w3), 12),
            bool(use_bn),
        )

        if key in done:
            skipped += 1
            continue

        rec = run_combo(
            x_train=x_train,
            x_val=x_val,
            y_train=y_train,
            y_val=y_val,
            lr=lr,
            label_smoothing=sm,
            l2_reg=l2_reg,
            w1=args.w1,
            w2=w2,
            w3=w3,
            use_bn=use_bn,
            batch_size=args.batch,
            epochs=args.epochs,
        )

        executed += 1
        pd.DataFrame([rec]).to_csv(
            args.results_csv,
            mode="a",
            header=not os.path.exists(args.results_csv),
            index=False,
        )
        print(f"[SWEEP] {rec}")

    print(f"\n[GRID] Saltadas: {skipped}")
    print(f"[GRID] Ejecutadas en esta corrida: {executed}\n")

    df_all = pd.read_csv(args.results_csv)
    df_all = df_all.sort_values("val_macroF1", ascending=False).reset_index(drop=True)
    best = df_all.iloc[0].to_dict()

    print("\n=== TOP-10 por Macro-F1 (Validación) ===")
    print(df_all.head(10).to_string(index=False))

    return best, df_all