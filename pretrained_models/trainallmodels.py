import os
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import build_model
from utils import EarlyStopping, eval_one_epoch, text_confusion_matrix, train_one_epoch


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_feature_extractor(model: nn.Module, nombre_modelo: str) -> nn.Module:
    nombre = nombre_modelo.lower()

    if "resnet" in nombre:
        backbone = nn.Sequential(*list(model.children())[:-1])
        return nn.Sequential(backbone, nn.Flatten(1))

    if "mobilenet" in nombre and hasattr(model, "features"):
        return nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
        )

    raise ValueError(f"No se pudo construir extractor para: {nombre_modelo}")


def preprocess_cnn_from_pil_original_size(img_rgb: Image.Image) -> torch.Tensor:
    x = np.asarray(img_rgb, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0)
    return x


def compute_stats(times_sec):
    if not times_sec:
        return {
            "mean_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "std_ms": 0.0,
            "mean_s": 0.0,
            "min_s": 0.0,
            "max_s": 0.0,
            "std_s": 0.0,
        }

    arr = np.array(times_sec, dtype=np.float64)
    mean_s = float(arr.mean())
    min_s = float(arr.min())
    max_s = float(arr.max())
    std_s = float(arr.std(ddof=0))

    return {
        "mean_s": mean_s,
        "min_s": min_s,
        "max_s": max_s,
        "std_s": std_s,
        "mean_ms": mean_s * 1000.0,
        "min_ms": min_s * 1000.0,
        "max_ms": max_s * 1000.0,
        "std_ms": std_s * 1000.0,
    }


def medir_tiempos_val_por_imagen(model: nn.Module, nombre_modelo: str, val_loader, device, out_csv_path: str):
    val_dataset = val_loader.dataset

    if hasattr(val_dataset, "samples"):
        val_paths = [sample[0] for sample in val_dataset.samples]
    elif hasattr(val_dataset, "imgs"):
        val_paths = [sample[0] for sample in val_dataset.imgs]
    else:
        raise RuntimeError(
            "No fue posible obtener las rutas de imágenes del conjunto de validación."
        )

    feature_extractor = build_feature_extractor(model, nombre_modelo).to(device)
    feature_extractor.eval()
    model.eval()

    times_feat = []
    times_inf = []
    rows = []

    with torch.no_grad():
        for path in val_paths:
            try:
                img_rgb = Image.open(path).convert("RGB")

                t0 = time.perf_counter()
                x = preprocess_cnn_from_pil_original_size(img_rgb).to(device)
                _ = feature_extractor(x)
                t1 = time.perf_counter()
                dt_feat = float(t1 - t0)

                t2 = time.perf_counter()
                x2 = preprocess_cnn_from_pil_original_size(img_rgb).to(device)
                _ = model(x2)
                t3 = time.perf_counter()
                dt_inf = float(t3 - t2)

                times_feat.append(dt_feat)
                times_inf.append(dt_inf)

                rows.append({
                    "filename": os.path.basename(path),
                    "path": path,
                    "H": img_rgb.size[1],
                    "W": img_rgb.size[0],
                    "time_features_sec": dt_feat,
                    "time_inference_sec": dt_inf,
                    "time_features_ms": dt_feat * 1000.0,
                    "time_inference_ms": dt_inf * 1000.0,
                })
            except Exception:
                continue

    try:
        import csv

        with open(out_csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "filename",
                    "path",
                    "H",
                    "W",
                    "time_features_sec",
                    "time_inference_sec",
                    "time_features_ms",
                    "time_inference_ms",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    except Exception:
        pass

    stats_feat = compute_stats(times_feat)
    stats_inf = compute_stats(times_inf)

    return {
        "val_count_measured": int(len(times_feat)),
        "val_times_csv": out_csv_path,
        "cnn_feat_mean_ms": float(stats_feat["mean_ms"]),
        "cnn_feat_min_ms": float(stats_feat["min_ms"]),
        "cnn_feat_max_ms": float(stats_feat["max_ms"]),
        "cnn_feat_std_ms": float(stats_feat["std_ms"]),
        "cnn_inf_mean_ms": float(stats_inf["mean_ms"]),
        "cnn_inf_min_ms": float(stats_inf["min_ms"]),
        "cnn_inf_max_ms": float(stats_inf["max_ms"]),
        "cnn_inf_std_ms": float(stats_inf["std_ms"]),
    }


def entrenar_modelo(nombre_modelo, data_dir, epochs=100, batch_size=32, lr=1e-4, patience=5):
    device = torch.device("cpu")
    print(f"\nEntrenando modelo: {nombre_modelo} en {device}")

    train_loader, val_loader, clases = get_dataloaders(data_dir=data_dir, batch_size=batch_size)
    model = build_model(nombre_modelo, num_classes=len(clases)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience, mode="min", min_delta=0.0)

    start_time = time.time()
    historia = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, _, _ = eval_one_epoch(model, val_loader, criterion, device)
        historia.append((epoch, train_loss, val_loss, val_acc, val_f1))

        print(
            f"Epoch {epoch:03d}: "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val Acc={val_acc:.4f} | "
            f"Val F1(macro)={val_f1:.4f}"
        )

        stop = early_stopping.step(val_loss, model, epoch)
        if stop:
            print(f"Early stopping activado (patience={patience}).")
            break

    early_stopping.load_best_weights(model, device)

    best_epoch = early_stopping.best_epoch
    best_val_loss = early_stopping.best
    final_val_loss, final_val_acc, final_val_f1, cm_norm, _ = eval_one_epoch(
        model,
        val_loader,
        criterion,
        device,
    )

    tiempos_csv = f"tiempos_val_{nombre_modelo}.csv"
    tiempos_val = medir_tiempos_val_por_imagen(
        model=model,
        nombre_modelo=nombre_modelo,
        val_loader=val_loader,
        device=device,
        out_csv_path=tiempos_csv,
    )

    best_path = f"best_{nombre_modelo}.pth"
    torch.save(model.state_dict(), best_path)

    total_time = time.time() - start_time
    minutos = int(total_time // 60)
    segundos = int(total_time % 60)

    print(f"\nMejor época: {best_epoch} | Mejor val_loss: {best_val_loss:.4f}")
    print(f"Mejores pesos guardados en: {best_path}")
    print(f"Tiempo total de entrenamiento {nombre_modelo}: {minutos} min {segundos} s")

    print("\nTiempos en validación por imagen (CPU, tamaño original)")
    print(
        f"CNN-total-features: mean={tiempos_val['cnn_feat_mean_ms']:.4f} ms | "
        f"min={tiempos_val['cnn_feat_min_ms']:.4f} ms | "
        f"max={tiempos_val['cnn_feat_max_ms']:.4f} ms | "
        f"std={tiempos_val['cnn_feat_std_ms']:.4f} ms"
    )
    print(
        f"Inferencia end-to-end: mean={tiempos_val['cnn_inf_mean_ms']:.4f} ms | "
        f"min={tiempos_val['cnn_inf_min_ms']:.4f} ms | "
        f"max={tiempos_val['cnn_inf_max_ms']:.4f} ms | "
        f"std={tiempos_val['cnn_inf_std_ms']:.4f} ms"
    )

    print(f"\nCSV con tiempos por imagen: {tiempos_val['val_times_csv']}")
    print("\nMatriz de confusión normalizada (filas = clase real):")
    print(text_confusion_matrix(cm_norm, labels=clases))

    print("Métricas finales con mejores pesos:")
    print(f"Val Loss = {final_val_loss:.4f}")
    print(f"Val Acc = {final_val_acc:.4f}")
    print(f"Val F1(macro) = {final_val_f1:.4f}")

    return {
        "historia": historia,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "final_val_loss": float(final_val_loss),
        "final_val_acc": float(final_val_acc),
        "final_val_f1": float(final_val_f1),
        "cm_norm": cm_norm,
        "checkpoint": best_path,
        "tiempo": f"{minutos} min {segundos} s",
        "clases": clases,
        **tiempos_val,
    }


def guardar_resultados(nombre_modelo, resumen, ruta="resultados_entrenamiento.txt"):
    with open(ruta, "a", encoding="utf-8") as file:
        file.write(f"\n### Resultados: {nombre_modelo} ###\n")
        file.write("Epoch | TrainLoss | ValLoss | ValAcc | ValF1(macro)\n")

        for epoch, train_loss, val_loss, val_acc, val_f1 in resumen["historia"]:
            file.write(
                f"{epoch:03d}   | {train_loss:.4f}    | {val_loss:.4f}  | "
                f"{val_acc:.4f}  | {val_f1:.4f}\n"
            )

        file.write(
            f"\nMejor época: {resumen['best_epoch']} "
            f"(val_loss={resumen['best_val_loss']:.4f})\n"
        )
        file.write(
            f"Final (mejores pesos): val_loss={resumen['final_val_loss']:.4f} | "
            f"val_acc={resumen['final_val_acc']:.4f} | "
            f"val_f1={resumen['final_val_f1']:.4f}\n"
        )
        file.write(f"Checkpoint: {resumen['checkpoint']}\n")

        file.write("Matriz de confusión normalizada (filas = real, cols = pred):\n")
        cm = resumen["cm_norm"]
        clases = resumen["clases"]
        width = 7
        header = "Pred\\True     | " + "  ".join([f"{c:>{width}}" for c in clases])
        file.write(header + "\n")
        file.write("-" * len(header) + "\n")

        for i, clase in enumerate(clases):
            row = "  ".join([f"{cm[i, j]:>{width}.2f}" for j in range(len(clases))])
            file.write(f"{clase:>12} | {row}\n")

        file.write(f"\nTiempo total: {resumen['tiempo']}\n")
        file.write("\nTiempos en validación (por imagen, CPU, tamaño original)\n")
        file.write("CNN-total-features (imagen cargada → normalización → forward extractor → vector features)\n")
        file.write(f"Imágenes medidas: {resumen.get('val_count_measured', 0)}\n")
        file.write(
            f"mean(ms)={resumen.get('cnn_feat_mean_ms', 0.0):.6f} | "
            f"min(ms)={resumen.get('cnn_feat_min_ms', 0.0):.6f} | "
            f"max(ms)={resumen.get('cnn_feat_max_ms', 0.0):.6f} | "
            f"std(ms)={resumen.get('cnn_feat_std_ms', 0.0):.6f}\n"
        )
        file.write("Inferencia end-to-end (imagen cargada → normalización → forward modelo completo → logits)\n")
        file.write(
            f"mean(ms)={resumen.get('cnn_inf_mean_ms', 0.0):.6f} | "
            f"min(ms)={resumen.get('cnn_inf_min_ms', 0.0):.6f} | "
            f"max(ms)={resumen.get('cnn_inf_max_ms', 0.0):.6f} | "
            f"std(ms)={resumen.get('cnn_inf_std_ms', 0.0):.6f}\n"
        )
        file.write(f"CSV tiempos por imagen: {resumen.get('val_times_csv', '')}\n")


def pedir_modelo_a_incluir(modelos):
    print("\nModelos disponibles:")
    for i, modelo in enumerate(modelos, 1):
        print(f"  {i}. {modelo}")

    while True:
        resp = input(
            "\nEscribe el nombre del modelo que quieres entrenar "
            "(Enter para entrenar todos): "
        ).strip()

        if resp == "":
            print("Se entrenarán todos los modelos.")
            return None

        candidatos = {modelo.lower(): modelo for modelo in modelos}
        if resp.lower() in candidatos:
            elegido = candidatos[resp.lower()]
            print(f"Se entrenará únicamente el modelo: {elegido}")
            return elegido

        print("Entrada no válida. Escribe uno de los nombres mostrados o presiona Enter.")


if __name__ == "__main__":
    modelos = ["mobilenetv2", "mobilenetv3", "resnet18"]
    data_dir = "data"
    epochs = 100
    batch_size = 32
    lr = 1e-4
    patience = 5

    if os.path.exists("resultados_entrenamiento.txt"):
        os.remove("resultados_entrenamiento.txt")

    modelo_a_incluir = pedir_modelo_a_incluir(modelos)

    for modelo in modelos:
        if modelo_a_incluir is not None and modelo != modelo_a_incluir:
            print(f"\nSaltando entrenamiento de: {modelo}")
            continue

        resumen = entrenar_modelo(
            nombre_modelo=modelo,
            data_dir=data_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
        )
        guardar_resultados(modelo, resumen)

    print("\nEntrenamiento finalizado. Revisa: resultados_entrenamiento.txt")