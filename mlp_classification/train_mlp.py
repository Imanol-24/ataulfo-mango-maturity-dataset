import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from config import build_parser
from data_utils import (
    class_weight_to_sample_weight,
    load_four_csvs,
    rebuild_split_indices,
    set_seed,
    split_and_scale_data,
)
from model_utils import build_multiclass_mlp, make_callbacks, make_optimizer
from repro_utils import file_sha256, save_repro_bundle
from sweep_utils import run_sweep


tf.get_logger().setLevel("ERROR")


def to_python_value(value):
    import numpy as _np
    if isinstance(value, (_np.floating, _np.integer, _np.bool_)):
        return value.item()
    return value


def train_best_model(args, best, x_train, x_val, x_test, y_train, y_val, y_test):
    num_classes = 4

    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)

    final_model = build_multiclass_mlp(
        input_dim=x_train.shape[1],
        arch="WIDE-512-256",
        act="leakyrelu",
        dropout=(0.25, 0.25),
        l2_reg=float(best["l2_reg"]),
        init_name="variancescaling",
        num_classes=num_classes,
        use_bn=bool(best["use_bn"]),
    )

    optimizer = make_optimizer(lr=float(best["lr"]))
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=float(best["label_smoothing"]))
    final_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
    )

    cw = {
        0: 1.0,
        1: float(best["w1"]),
        2: float(best["w2"]),
        3: float(best["w3"]),
    }
    sw_train = class_weight_to_sample_weight(y_train, cw)

    callbacks = make_callbacks("clasico_best.keras")

    final_model.fit(
        x_train,
        y_train_oh,
        validation_data=(x_val, y_val_oh),
        epochs=int(args.epochs),
        batch_size=int(best["batch"]),
        callbacks=callbacks,
        verbose=0,
        sample_weight=sw_train,
    )

    y_prob = final_model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])

    out = {
        "seed": args.seed,
        "best_grid_row_val": best,
        "test_macroF1": float(macro_f1),
        "report": report,
        "confusion_matrix_labels": [0, 1, 2, 3],
        "confusion_matrix": cm.tolist(),
    }

    with open("clasico_best_results.json", "w", encoding="utf-8") as file:
        json.dump(out, file, indent=2, ensure_ascii=False)

    print("\n=== RESULTADO FINAL (TEST) ===")
    print(json.dumps(out, indent=2, ensure_ascii=False))

    final_model.save("clasico_best.keras")

    return final_model, report, macro_f1, cm, y_pred


def main():
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    x, y, used_cols = load_four_csvs(
        args.file_r01,
        args.file_r02,
        args.file_r03,
        args.file_r04,
    )

    x_train, x_val, x_test, y_train, y_val, y_test, scaler = split_and_scale_data(
        x=x,
        y=y,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    best, df_all = run_sweep(args, x_train, x_val, y_train, y_val)
    best_py = {k: to_python_value(v) for k, v in best.items()}

    _, report, macro_f1, cm, y_pred = train_best_model(
        args=args,
        best=best_py,
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    try:
        split_indices = rebuild_split_indices(
            y=y,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
        )

        data_info = {
            "R01": {"path": args.file_r01, "sha256": file_sha256(args.file_r01)},
            "R02": {"path": args.file_r02, "sha256": file_sha256(args.file_r02)},
            "R03": {"path": args.file_r03, "sha256": file_sha256(args.file_r03)},
            "R04": {"path": args.file_r04, "sha256": file_sha256(args.file_r04)},
        }

        extra = {}
        for name in ("clasico_best.keras", "clasico_sweep_results.csv", "clasico_best_results.json"):
            if os.path.exists(name):
                extra[name] = name

        bundle_name = f"repro_bundle_clasico_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        save_repro_bundle(
            bundle_dir=bundle_name,
            args=args,
            data_info=data_info,
            split_indices=split_indices,
            scaler=scaler,
            best_cfg=best_py,
            val_macro_f1=best_py.get("val_macroF1", None),
            test_report=report,
            test_macro_f1=macro_f1,
            cm=cm,
            y_test=y_test,
            y_pred_test=y_pred,
            used_cols=used_cols,
            extra_files=extra,
        )

        print(f"\n[REPRO] Paquete de replicación guardado en: {bundle_name}")

    except Exception as error:
        print(f"[REPRO][WARN] No se pudo crear el paquete de replicación: {error}")


if __name__ == "__main__":
    main()