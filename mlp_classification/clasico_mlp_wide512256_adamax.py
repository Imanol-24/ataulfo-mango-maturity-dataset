import os, json, argparse, itertools
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import VarianceScaling, HeNormal

tf.get_logger().setLevel("ERROR")

# === ADICIONES PARA REPLICABILIDAD (solo utilidades/IO; NO afecta entrenamiento) ===
import joblib, hashlib, platform, sys, subprocess, shutil
from datetime import datetime
from typing import Dict, Any

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def get_environment_info() -> Dict[str, Any]:
    build = tf.sysconfig.get_build_info() if hasattr(tf.sysconfig, "get_build_info") else {}
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_sha = None
    try:
        import numpy as _np; np_ver = _np.__version__
    except Exception:
        np_ver = None
    try:
        import pandas as _pd; pd_ver = _pd.__version__
    except Exception:
        pd_ver = None
    try:
        import sklearn as _skl; skl_ver = _skl.__version__
    except Exception:
        skl_ver = None
    gpus = []
    try:
        gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception:
        pass
    return {
        "python": sys.version,
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "libraries": {"tensorflow": tf.__version__, "numpy": np_ver, "pandas": pd_ver, "scikit_learn": skl_ver, "joblib": getattr(joblib, "__version__", None)},
        "tf_build": build,
        "cuda_built": getattr(tf.test, "is_built_with_cuda", lambda: None)(),
        "visible_gpus": gpus,
        "env_flags": {
            "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
            "TF_CUDNN_DETERMINISTIC": os.environ.get("TF_CUDNN_DETERMINISTIC"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")
        }
    }

def save_repro_bundle(
    bundle_dir: str,
    args,
    data_info: Dict[str, Any],
    split_indices: Dict[str, np.ndarray],
    scaler,
    best_cfg: Dict[str, Any],
    threshold_parent: float,
    val_macroF1: float,
    test_report: Dict[str, Any],
    test_macroF1: float,
    cm: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    extra_files: Dict[str, str] = None
):
    ensure_dir(bundle_dir)
    artifacts_dir = os.path.join(bundle_dir, "artifacts")
    ensure_dir(artifacts_dir)

    if scaler is not None:
        joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))
    if split_indices:
        np.save(os.path.join(artifacts_dir, "idx_train.npy"), split_indices.get("train"))
        np.save(os.path.join(artifacts_dir, "idx_val.npy"),   split_indices.get("val"))
        np.save(os.path.join(artifacts_dir, "idx_test.npy"),  split_indices.get("test"))

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
            "strategy": "stratified_holdout_same_seed"
        },
        "preprocessing": {
            "scaler": type(scaler).__name__ if scaler is not None else None,
            "feature_range": feature_range,
            "used_feature_columns": globals().get("used_cols", [])
        },
        "search_selection": {
            "metric": "val_macroF1",
            "best_val_macroF1": float(val_macroF1) if val_macroF1 is not None else None,
            "best_threshold_parent": None
        },
        "winning_config": best_cfg,
        "test_results": {"macroF1": float(test_macroF1) if test_macroF1 is not None else None, "classification_report": test_report},
        "environment": get_environment_info(),
        "command_line": " ".join(sys.argv)
    }
    with open(os.path.join(bundle_dir, "repro_config.json"), "w", encoding="utf-8") as f:
        json.dump(repro_config, f, indent=2, ensure_ascii=False)

    readme = """REPRODUCIBILIDAD — Paquete del mejor modelo (MLP multiclase)

1) Entorno:
   - Versiones y GPU: repro_config.json -> environment.

2) Datos:
   - Rutas + hashes SHA-256: repro_config.json -> data.
   - Columnas usadas: repro_config.json -> preprocessing.used_feature_columns.
   - Splits exactos: artifacts/idx_train.npy, idx_val.npy, idx_test.npy.

3) Preprocesamiento:
   - Scaler entrenado: artifacts/scaler.pkl (MinMaxScaler(-1,1)).

4) Modelo:
   - Guardado en artifacts/ (clasico_best.keras).
   - Hiperparámetros ganadores: repro_config.json -> winning_config.

5) Métricas:
   - F1-macro test: repro_config.json -> test_results.macroF1
   - Reporte completo y matriz de confusión (CSV).

6) Búsqueda:
   - clasico_sweep_results.csv con todas las combinaciones y su score.
"""
    with open(os.path.join(bundle_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(readme)
# === FIN ADICIONES ===


# ---------- RUTAS POR DEFECTO (los mismos CSV) ----------
DEFAULT_R01 = r"/home/imanol/Documentos/Cuarto_Semestre/nuevoCarallDes/caracteristicas_mango_allDes_R01.csv"  # Clase 0
DEFAULT_R02 = r"/home/imanol/Documentos/Cuarto_Semestre/nuevoCarallDes/caracteristicas_mango_allDes_R02.csv"  # Clase 1
DEFAULT_R03 = r"/home/imanol/Documentos/Cuarto_Semestre/nuevoCarallDes/caracteristicas_mango_allDes_R03.csv"  # Clase 2
DEFAULT_R04 = r"/home/imanol/Documentos/Cuarto_Semestre/nuevoCarallDes/caracteristicas_mango_allDes_R04.csv"  # Clase 3

# ---------- Utils ----------
def set_seed(seed=1337):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def _read_numeric_csv(path):
    df = pd.read_csv(path)
    df = df.select_dtypes(include=[np.number]).copy()
    df = df.fillna(0)
    return df

def load_four_csvs(file_r01, file_r02, file_r03, file_r04):
    d1, d2, d3, d4 = map(_read_numeric_csv, [file_r01, file_r02, file_r03, file_r04])
    common_cols = sorted(list(set(d1.columns) & set(d2.columns) & set(d3.columns) & set(d4.columns)))
    if not common_cols:
        raise ValueError("No hay columnas numéricas comunes entre los CSV.")
    X0, X1, X2, X3 = d1[common_cols].values, d2[common_cols].values, d3[common_cols].values, d4[common_cols].values
    y0 = np.zeros((len(X0),), dtype=np.int32)
    y1 = np.ones((len(X1),), dtype=np.int32)
    y2 = np.full((len(X2),), 2, dtype=np.int32)
    y3 = np.full((len(X3),), 3, dtype=np.int32)
    X = np.vstack([X0, X1, X2, X3]).astype("float32")
    y = np.concatenate([y0, y1, y2, y3]).astype("int32")
    return X, y, common_cols

def class_weight_to_sample_weight(y, cw_map):
    return np.asarray([cw_map[int(c)] for c in y], dtype="float32")

# ---------- Arquitecturas ----------
def act_layer(name):
    name = (name or "leakyrelu").lower()
    if name == "softsign":
        return tf.keras.layers.Activation("softsign")
    return tf.keras.layers.LeakyReLU(negative_slope=0.1)

def dense_block(units, act="leakyrelu", use_bn=True, dropout=0.0, l2_reg=0.0, init_name="variancescaling"):
    kernel_init = VarianceScaling() if init_name == "variancescaling" else HeNormal()
    layers = []
    layers.append(Dense(
        units,
        use_bias=not use_bn,
        kernel_initializer=kernel_init,
        kernel_regularizer=l2(l2_reg) if l2_reg and l2_reg > 0 else None
    ))
    if use_bn:
        layers.append(BatchNormalization())
    layers.append(act_layer(act))
    if dropout and dropout > 0:
        layers.append(Dropout(dropout))
    return layers

def build_multiclass_mlp(
    input_dim,
    arch="WIDE-512-256",
    act="leakyrelu",
    dropout=(0.25, 0.25),
    l2_reg=3e-05,
    init_name="variancescaling",
    num_classes=4,
    use_bn=True
):
    arch = arch.upper()
    model = Sequential(name=f"MLP_{arch}")

    if arch != "WIDE-512-256":
        raise ValueError(f"Este script está configurado para WIDE-512-256. Recibido: {arch}")

    if isinstance(dropout, (list, tuple)) and len(dropout) == 2:
        d1, d2 = float(dropout[0]), float(dropout[1])
    else:
        base = max(float(dropout), 0.25)
        d1 = d2 = base

    for layer in dense_block(512, act=act, use_bn=use_bn, dropout=max(d1, 0.25), l2_reg=l2_reg, init_name=init_name):
        model.add(layer)
    for layer in dense_block(256, act=act, use_bn=use_bn, dropout=max(d2, 0.25), l2_reg=l2_reg, init_name=init_name):
        model.add(layer)

    model.add(Dense(num_classes, activation="softmax"))
    model.build((None, input_dim))
    return model

# ---------- Optimizador ----------
def make_optimizer(lr=3e-3):
    return tf.keras.optimizers.Adam(learning_rate=lr)

# ---------- Entrena/Evalúa una combinación ----------
def run_combo(
    X_train, X_val, y_train, y_val,
    lr, label_smoothing, l2_reg,
    w1, w2, w3,
    use_bn,
    batch_size=128,
    epochs=120
):
    tf.keras.backend.clear_session()

    arch = "WIDE-512-256"
    act  = "leakyrelu"
    init_fixed = "variancescaling"
    dropout_for_model = (0.25, 0.25)
    num_classes = 4

    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   num_classes=num_classes)

    model = build_multiclass_mlp(
        input_dim=X_train.shape[1],
        arch=arch,
        act=act,
        dropout=dropout_for_model,
        l2_reg=float(l2_reg),
        init_name=init_fixed,
        num_classes=num_classes,
        use_bn=bool(use_bn)
    )

    opt = make_optimizer(lr=float(lr))
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=float(label_smoothing))
    model.compile(optimizer=opt, loss=loss, metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])

    cw = {0: 1.0, 1: float(w1), 2: float(w2), 3: float(w3)}
    sw_train = class_weight_to_sample_weight(y_train, cw)

    ckpt = "tmp_clasico_best.keras"
    cbs = [
        ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=0),
        EarlyStopping(monitor="val_loss", patience=18, restore_best_weights=True, verbose=0),
    ]

    model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=int(epochs),
        batch_size=int(batch_size),
        callbacks=cbs,
        verbose=0,
        sample_weight=sw_train
    )

    try:
        model = tf.keras.models.load_model(ckpt, compile=False)
    except Exception:
        pass

    y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    val_macroF1 = f1_score(y_val, y_val_pred, average="macro")

    rec = {
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
        "val_macroF1": float(val_macroF1)
    }
    return rec

def _boolish(x: str) -> bool:
    s = str(x).strip().lower()
    return s in ("true","1","yes","y","t")

def _key_from_row(r: pd.Series):
    return (
        round(float(r["lr"]), 12),
        round(float(r["label_smoothing"]), 12),
        round(float(r["l2_reg"]), 12),
        round(float(r["w2"]), 12),
        round(float(r["w3"]), 12),
        bool(r["use_bn"]),
    )

# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser()

    # CSVs
    ap.add_argument("--file_r01", default=DEFAULT_R01)
    ap.add_argument("--file_r02", default=DEFAULT_R02)
    ap.add_argument("--file_r03", default=DEFAULT_R03)
    ap.add_argument("--file_r04", default=DEFAULT_R04)

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--test_size", type=float, default=0.15)

    # === Sweep (576 combinaciones por default) ===
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr_list", default="0.003,0.005")                 # 2
    ap.add_argument("--smooth_list", default="0.02,0.015,0.01")         # 3
    ap.add_argument("--l2_list", default="1e-5,3e-5,1e-4")              # 3
    ap.add_argument("--w1", type=float, default=1.2)                    # fijo
    ap.add_argument("--w2_list", default="1.2,1.3,1.4,1.5")             # 4
    ap.add_argument("--w3_list", default="1.2,1.3,1.4,1.5")             # 4
    ap.add_argument("--bn_list", default="true,false")                  # 2

    ap.add_argument("--epochs", type=int, default=120)

    # === Reanudar ===
    ap.add_argument("--resume", action="store_true", help="Si se activa, reanuda el sweep usando clasico_sweep_results.csv")
    ap.add_argument("--results_csv", default="clasico_sweep_results.csv")

    args = ap.parse_args()
    set_seed(args.seed)

    # Carga y split
    X, y, used_cols = load_four_csvs(args.file_r01, args.file_r02, args.file_r03, args.file_r04)
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y,
        test_size=(args.val_size + args.test_size),
        stratify=y,
        random_state=args.seed
    )
    rel_test = args.test_size / (args.val_size + args.test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest,
        test_size=rel_test,
        stratify=y_rest,
        random_state=args.seed
    )

    # Escalado sin fuga
    scaler = MinMaxScaler(feature_range=(-1,1))
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Parse lists
    lrs     = [float(s) for s in args.lr_list.split(",") if s.strip()]
    smooths = [float(s) for s in args.smooth_list.split(",") if s.strip()]
    l2s     = [float(s) for s in args.l2_list.split(",") if s.strip()]
    w2s     = [float(s) for s in args.w2_list.split(",") if s.strip()]
    w3s     = [float(s) for s in args.w3_list.split(",") if s.strip()]
    bn_raw  = [s.strip() for s in args.bn_list.split(",") if s.strip()]
    bns     = [_boolish(s) for s in bn_raw]

    expected_total = len(lrs) * len(smooths) * len(l2s) * len(w2s) * len(w3s) * len(bns)
    print(f"\n[GRID] Combinaciones esperadas: {expected_total} (576 si usas defaults)\n")

    results_csv = args.results_csv

    # ===== RESUME: cargar combinaciones ya hechas =====
    done = set()
    if args.resume and os.path.exists(results_csv):
        prev = pd.read_csv(results_csv)
        for _, r in prev.iterrows():
            try:
                done.add(_key_from_row(r))
            except Exception:
                pass
        print(f"[RESUME] Encontradas {len(done)} combinaciones ya guardadas en {results_csv}. Reanudando…\n")
    elif args.resume:
        print(f"[RESUME] --resume activado, pero no existe {results_csv}. Iniciando desde cero.\n")
    else:
        # si NO es resume, reiniciar archivo
        if os.path.exists(results_csv):
            os.remove(results_csv)
        print("[RESUME] Modo normal: se reinicia el sweep (se borra CSV previo si existe).\n")

    # Sweep
    rows = []
    ran = 0
    skipped = 0

    for lr, sm, l2_reg, w2, w3, use_bn in itertools.product(lrs, smooths, l2s, w2s, w3s, bns):
        key = (round(float(lr), 12), round(float(sm), 12), round(float(l2_reg), 12),
               round(float(w2), 12), round(float(w3), 12), bool(use_bn))
        if key in done:
            skipped += 1
            continue

        rec = run_combo(
            X_train, X_val, y_train, y_val,
            lr=lr,
            label_smoothing=sm,
            l2_reg=l2_reg,
            w1=args.w1,
            w2=w2,
            w3=w3,
            use_bn=use_bn,
            batch_size=args.batch,
            epochs=args.epochs
        )

        ran += 1
        rows.append(rec)
        pd.DataFrame([rec]).to_csv(results_csv, mode="a", header=not os.path.exists(results_csv), index=False)
        print(f"[SWEEP] {rec}")

    print(f"\n[GRID] Saltadas (ya existentes): {skipped}")
    print(f"[GRID] Ejecutadas en esta corrida: {ran}\n")

    # ===== Cargar TODO el CSV (incluyendo previos) para seleccionar el mejor =====
    df_all = pd.read_csv(results_csv)
    df_all = df_all.sort_values("val_macroF1", ascending=False).reset_index(drop=True)
    best = df_all.iloc[0].to_dict()

    print("\n=== TOP-10 por Macro-F1 (Validación) ===")
    print(df_all.head(10).to_string(index=False))

    # ===== Re-entrenar mejor config y evaluar en TEST =====
    num_classes = 4
    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   num_classes=num_classes)

    final_model = build_multiclass_mlp(
        input_dim=X_train.shape[1],
        arch="WIDE-512-256",
        act="leakyrelu",
        dropout=(0.25, 0.25),
        l2_reg=float(best["l2_reg"]),
        init_name="variancescaling",
        num_classes=num_classes,
        use_bn=bool(best["use_bn"])
    )
    opt = make_optimizer(lr=float(best["lr"]))
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=float(best["label_smoothing"]))
    final_model.compile(optimizer=opt, loss=loss, metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])

    cw = {0: 1.0, 1: float(best["w1"]), 2: float(best["w2"]), 3: float(best["w3"])}
    sw_train = class_weight_to_sample_weight(y_train, cw)

    cbs = [
        ModelCheckpoint("clasico_best.keras", monitor="val_loss", save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=0),
        EarlyStopping(monitor="val_loss", patience=18, restore_best_weights=True, verbose=0)
    ]
    final_model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=int(args.epochs),
        batch_size=int(best["batch"]),
        callbacks=cbs,
        verbose=0,
        sample_weight=sw_train
    )

    y_prob = final_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred, labels=[0,1,2,3])

    out = {
        "seed": args.seed,
        "used_feature_columns": used_cols,
        "best_grid_row_val": best,
        "test_macroF1": float(macro_f1),
        "report": report,
        "confusion_matrix_labels": [0,1,2,3],
        "confusion_matrix": cm.tolist()
    }
    with open("clasico_best_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n=== RESULTADO FINAL (TEST) ===")
    print(json.dumps(out, indent=2, ensure_ascii=False))

    final_model.save("clasico_best.keras")

    # === ADICIÓN FINAL: EMPAQUETADO DE REPLICACIÓN (no altera entrenamiento) ===
    try:
        n_total = len(y)
        all_idx = np.arange(n_total)

        idx_train, idx_rest = train_test_split(
            all_idx,
            test_size=(args.val_size + args.test_size),
            stratify=y,
            random_state=args.seed
        )
        y_rest2 = y[idx_rest]
        rel_test2 = args.test_size / (args.val_size + args.test_size)
        idx_val, idx_test = train_test_split(
            idx_rest,
            test_size=rel_test2,
            stratify=y_rest2,
            random_state=args.seed
        )
        split_indices = {"train": idx_train, "val": idx_val, "test": idx_test}

        data_info = {}
        for attr in ("file_r01","file_r02","file_r03","file_r04"):
            if hasattr(args, attr) and getattr(args, attr):
                path = getattr(args, attr)
                data_info[attr[-3:]] = {"path": path, "sha256": file_sha256(path)}

        extra = {}
        for name in ("clasico_best.keras","clasico_sweep_results.csv","clasico_best_results.json"):
            if os.path.exists(name):
                extra[name] = name

        def _to_py(v):
            import numpy as _np
            if isinstance(v, (_np.floating, _np.integer, _np.bool_)):
                return v.item()
            return v
        best_py = {k: _to_py(v) for k, v in (best.items() if isinstance(best, dict) else [])}

        bundle_name = f"repro_bundle_clasico_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        globals()["used_cols"] = used_cols

        save_repro_bundle(
            bundle_dir=bundle_name,
            args=args,
            data_info=data_info,
            split_indices=split_indices,
            scaler=scaler,
            best_cfg=best_py,
            threshold_parent=None,
            val_macroF1=best_py.get("val_macroF1", None),
            test_report=report,
            test_macroF1=macro_f1,
            cm=cm,
            y_test=y_test,
            y_pred_test=y_pred,
            extra_files=extra
        )
        print(f"\n[REPRO] Paquete de replicación guardado en: {bundle_name}")
    except Exception as e:
        print(f"[REPRO][WARN] No se pudo crear el paquete de replicación: {e}")
    # === FIN ADICIÓN ===

if __name__ == "__main__":
    main()
