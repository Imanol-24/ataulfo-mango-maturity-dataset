import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime

import joblib
import numpy as np
import tensorflow as tf


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_environment_info():
    build = tf.sysconfig.get_build_info() if hasattr(tf.sysconfig, "get_build_info") else {}

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_sha = None

    try:
        import numpy as _np
        np_ver = _np.__version__
    except Exception:
        np_ver = None

    try:
        import pandas as _pd
        pd_ver = _pd.__version__
    except Exception:
        pd_ver = None

    try:
        import sklearn as _skl
        skl_ver = _skl.__version__
    except Exception:
        skl_ver = None

    try:
        gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception:
        gpus = []

    return {
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "libraries": {
            "tensorflow": tf.__version__,
            "numpy": np_ver,
            "pandas": pd_ver,
            "scikit_learn": skl_ver,
            "joblib": getattr(joblib, "__version__", None),
        },
        "tf_build": build,
        "cuda_built": getattr(tf.test, "is_built_with_cuda", lambda: None)(),
        "visible_gpus": gpus,
        "env_flags": {
            "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
            "TF_CUDNN_DETERMINISTIC": os.environ.get("TF_CUDNN_DETERMINISTIC"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        },
    }


def save_repro_bundle(
    bundle_dir,
    args,
    data_info,
    split_indices,
    scaler,
    best_cfg,
    val_macro_f1,
    test_report,
    test_macro_f1,
    cm,
    y_test,
    y_pred_test,
    used_cols,
    extra_files=None,
):
    ensure_dir(bundle_dir)
    artifacts_dir = os.path.join(bundle_dir, "artifacts")
    ensure_dir(artifacts_dir)

    if scaler is not None:
        joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))

    if split_indices:
        np.save(os.path.join(artifacts_dir, "idx_train.npy"), split_indices.get("train"))
        np.save(os.path.join(artifacts_dir, "idx_val.npy"), split_indices.get("val"))
        np.save(os.path.join(artifacts_dir, "idx_test.npy"), split_indices.get("test"))

    if y_test is not None:
        np.save(os.path.join(artifacts_dir, "y_test.npy"), y_test)

    if y_pred_test is not None:
        np.save(os.path.join(artifacts_dir, "y_pred_test.npy"), y_pred_test)

    if cm is not None:
        np.savetxt(os.path.join(artifacts_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    for src, target in (extra_files or {}).items():
        if src and os.path.exists(src):
            shutil.copy2(src, os.path.join(artifacts_dir, target))

    feature_range = None
    try:
        if hasattr(scaler, "feature_range"):
            feature_range = tuple(scaler.feature_range)
    except Exception:
        pass

    repro_config = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": getattr(args, "seed", None),
        "data": data_info,
        "splits": {
            "val_size": getattr(args, "val_size", None),
            "test_size": getattr(args, "test_size", None),
            "strategy": "stratified_holdout_same_seed",
        },
        "preprocessing": {
            "scaler": type(scaler).__name__ if scaler is not None else None,
            "feature_range": feature_range,
            "used_feature_columns": used_cols,
        },
        "search_selection": {
            "metric": "val_macroF1",
            "best_val_macroF1": float(val_macro_f1) if val_macro_f1 is not None else None,
        },
        "winning_config": best_cfg,
        "test_results": {
            "macroF1": float(test_macro_f1) if test_macro_f1 is not None else None,
            "classification_report": test_report,
        },
        "environment": get_environment_info(),
        "command_line": " ".join(sys.argv),
    }

    with open(os.path.join(bundle_dir, "repro_config.json"), "w", encoding="utf-8") as file:
        json.dump(repro_config, file, indent=2, ensure_ascii=False)

    readme = """REPRODUCIBILIDAD — Paquete del mejor modelo (MLP multiclase)

1) Entorno:
   - Versiones y GPU: repro_config.json -> environment.

2) Datos:
   - Rutas + hashes SHA-256: repro_config.json -> data.
   - Columnas usadas: repro_config.json -> preprocessing.used_feature_columns.
   - Splits exactos: artifacts/idx_train.npy, idx_val.npy, idx_test.npy.

3) Preprocesamiento:
   - Scaler entrenado: artifacts/scaler.pkl.

4) Modelo:
   - Guardado en artifacts/.
   - Hiperparámetros ganadores: repro_config.json -> winning_config.

5) Métricas:
   - F1-macro test: repro_config.json -> test_results.macroF1
   - Reporte completo y matriz de confusión.

6) Búsqueda:
   - clasico_sweep_results.csv con todas las combinaciones y su score.
"""
    with open(os.path.join(bundle_dir, "README.txt"), "w", encoding="utf-8") as file:
        file.write(readme)