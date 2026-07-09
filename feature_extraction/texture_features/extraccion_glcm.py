import os
import re
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


IMAGES_DIRS = [
    'ruta/a/tu/carpeta/R01',
    'ruta/a/tu/carpeta/R02',
    'ruta/a/tu/carpeta/R03',
    'ruta/a/tu/carpeta/R04'
]
OUTPUT_DIR = 'ruta/a/tu/carpeta_de_salida'
CLASS_MAP_TOKENS = {"R01": 1, "R02": 2, "R03": 3, "R04": 4}

LEVELS = 32
DIST = 1
ANGLES_DEG = (0, 45, 90, 135)
MASK_THRESHOLD = 0.0


@dataclass
class Config:
    images_dirs: List[str]
    output_dir: str
    class_map_tokens: Dict[str, int]
    valid_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_class_map_tokens(mapping: Dict[str, int]) -> Dict[str, int]:
    return {str(k).strip().lower(): int(v) for k, v in mapping.items()}


def infer_token_from_path(root: str) -> str:
    root = os.path.normpath(root)
    base = os.path.basename(root)
    parent = os.path.basename(os.path.dirname(root))
    return parent if parent else base


def auto_numeric_from_token(token: str) -> Optional[int]:
    m = re.match(r"^[Rr]\s*0*([1-9]\d*)$", token.strip())
    return int(m.group(1)) if m else None


def list_images_in_root(root: str, valid_exts: Tuple[str, ...]) -> List[str]:
    try:
        return [
            os.path.join(root, f) for f in sorted(os.listdir(root))
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
    except FileNotFoundError:
        return []


def load_gray_u8(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise IOError(f"No se pudo leer la imagen: {path}")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return gray


def quantize_gray_u8_to_levels(gray8: np.ndarray, levels: int = LEVELS) -> np.ndarray:
    q = np.floor(gray8.astype(np.float32) * (levels / 256.0)).astype(np.int32)
    q[q < 0] = 0
    q[q >= levels] = levels - 1
    return q.astype(np.uint8)


def angle_to_offset(angle_deg: int, dist: int) -> Tuple[int, int]:
    if angle_deg == 0:
        return (dist, 0)
    if angle_deg == 45:
        return (dist, -dist)
    if angle_deg == 90:
        return (0, -dist)
    if angle_deg == 135:
        return (-dist, -dist)
    raise ValueError(f"Ángulo no soportado: {angle_deg}")


def masked_glcm(q: np.ndarray, mask: np.ndarray, levels: int, dist: int, angles_deg: Tuple[int, ...]) -> np.ndarray:
    height, width = q.shape
    glcm = np.zeros((levels, levels, 1, len(angles_deg)), dtype=np.uint64)

    for angle_index, angle in enumerate(angles_deg):
        dx, dy = angle_to_offset(angle, dist)

        ys_src = slice(max(0, dy), min(height, height + dy))
        xs_src = slice(max(0, dx), min(width, width + dx))
        ys_dst = slice(max(0, -dy), min(height, height - dy))
        xs_dst = slice(max(0, -dx), min(width, width - dx))

        source_pixels = q[ys_src, xs_src]
        destination_pixels = q[ys_dst, xs_dst]
        valid_mask = mask[ys_src, xs_src] & mask[ys_dst, xs_dst]

        if not np.any(valid_mask):
            continue

        source_values = source_pixels[valid_mask].ravel().astype(np.int64)
        destination_values = destination_pixels[valid_mask].ravel().astype(np.int64)
        indices = source_values * levels + destination_values
        counts = np.bincount(indices, minlength=levels * levels).reshape(levels, levels)
        glcm[:, :, 0, angle_index] = counts

    glcm = glcm + np.transpose(glcm, (1, 0, 2, 3))
    glcm = glcm.astype(np.float64)
    total = glcm.sum(axis=(0, 1), keepdims=True)
    total[total == 0] = 1.0
    glcm /= total
    return glcm


def glcm_props(glcm: np.ndarray) -> Dict[str, np.ndarray]:
    levels = glcm.shape[0]
    values = np.arange(levels, dtype=np.float64)
    grid_i, grid_j = np.meshgrid(values, values, indexing='ij')

    num_angles = glcm.shape[3]
    contrast = np.zeros((1, num_angles), dtype=np.float64)
    dissimilarity = np.zeros((1, num_angles), dtype=np.float64)
    homogeneity = np.zeros((1, num_angles), dtype=np.float64)
    energy = np.zeros((1, num_angles), dtype=np.float64)
    correlation = np.zeros((1, num_angles), dtype=np.float64)

    for angle_index in range(num_angles):
        probability = glcm[:, :, 0, angle_index]
        px = probability.sum(axis=1)
        py = probability.sum(axis=0)
        mean_x = (values * px).sum()
        mean_y = (values * py).sum()
        std_x = np.sqrt(((values - mean_x) ** 2 * px).sum())
        std_y = np.sqrt(((values - mean_y) ** 2 * py).sum())

        diff = grid_i - grid_j
        contrast[0, angle_index] = (probability * (diff ** 2)).sum()
        dissimilarity[0, angle_index] = (probability * np.abs(diff)).sum()
        homogeneity[0, angle_index] = (probability / (1.0 + diff ** 2)).sum()
        energy[0, angle_index] = (probability ** 2).sum()

        if std_x > 0 and std_y > 0:
            correlation[0, angle_index] = (
                (probability * (grid_i - mean_x) * (grid_j - mean_y)).sum()
            ) / (std_x * std_y)
        else:
            correlation[0, angle_index] = np.nan

    return {
        "contrast": contrast,
        "dissimilarity": dissimilarity,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
    }


def run_pipeline(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)

    token_to_label: Dict[str, int] = {}
    for root in cfg.images_dirs:
        if not os.path.isdir(root):
            print(f"[AVISO] No existe: {root}", file=sys.stderr)
            continue

        token = infer_token_from_path(root)
        key = token.lower()
        token_to_label[key] = cfg.class_map_tokens.get(key, auto_numeric_from_token(token) or 0)

    rows = []
    for root in cfg.images_dirs:
        if not os.path.isdir(root):
            continue

        token = infer_token_from_path(root)
        key = token.lower()
        label = token_to_label.get(key, 0)

        if label == 0:
            print(f"[AVISO] Token '{token}' sin clase numérica. Ajusta CLASS_MAP_TOKENS.", file=sys.stderr)
            continue

        img_paths = list_images_in_root(root, cfg.valid_exts)
        if not img_paths:
            print(f"[AVISO] Sin imágenes en: {root}", file=sys.stderr)
            continue

        for image_path in img_paths:
            try:
                gray = load_gray_u8(image_path)

                mask = (gray.astype(np.float32) / 255.0) > MASK_THRESHOLD
                region = gray.copy()
                region[~mask] = 0

                quantized = quantize_gray_u8_to_levels(region, levels=LEVELS)
                glcm = masked_glcm(quantized, mask, levels=LEVELS, dist=DIST, angles_deg=ANGLES_DEG)
                props = glcm_props(glcm)

                feats = {
                    "GLCM_contrast_mean": float(np.nanmean(props["contrast"][0, :])),
                    "GLCM_correlation_mean": float(np.nanmean(props["correlation"][0, :])),
                    "GLCM_energy_mean": float(np.nanmean(props["energy"][0, :])),
                    "GLCM_homogeneity_mean": float(np.nanmean(props["homogeneity"][0, :])),
                    "GLCM_dissimilarity_mean": float(np.nanmean(props["dissimilarity"][0, :])),
                }

                rows.append({
                    "filename": os.path.basename(image_path),
                    "class_token": token,
                    "label": int(label),
                    **feats
                })

            except Exception as error:
                print(f"[ERROR] {image_path}: {error}", file=sys.stderr)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No se extrajeron características. Verifica rutas.")

    out_csv = os.path.join(cfg.output_dir, "glcm_features_masked_mean.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] Guardado: {out_csv}")


if __name__ == "__main__":
    cfg = Config(
        images_dirs=IMAGES_DIRS,
        output_dir=OUTPUT_DIR,
        class_map_tokens=parse_class_map_tokens(CLASS_MAP_TOKENS),
    )
    run_pipeline(cfg)