import os
import sys
import re
import math
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
EPS = np.float32(1e-8)

IMAGES_DIRS = [
    'ruta/a/tu/carpeta/R01',
    'ruta/a/tu/carpeta/R02',
    'ruta/a/tu/carpeta/R03',
    'ruta/a/tu/carpeta/R04'
]
OUTPUT_DIR = 'ruta/a/tu/carpeta_de_salida'
CLASS_MAP_TOKENS = {"R01": 1, "R02": 2, "R03": 3, "R04": 4}
BLACK_THRESHOLD = 0


@dataclass
class Config:
    images_dirs: List[str]
    output_dir: str
    class_map_tokens: Dict[str, int]
    black_threshold: int
    valid_exts: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def parse_class_map_tokens(mapping: Dict[str, int]) -> Dict[str, int]:
    return {str(k).strip().lower(): int(v) for k, v in mapping.items()}


def infer_token_from_path(root: str) -> str:
    root = os.path.normpath(root)
    base = os.path.basename(root)
    parent = os.path.basename(os.path.dirname(root))
    token = parent if parent else base
    return token


def auto_numeric_from_token(token: str) -> Optional[int]:
    m = re.match(r'^[Rr]\s*0*([1-9]\d*)$', token.strip())
    if m:
        return int(m.group(1))
    return None


def list_images_in_root(root: str, valid_exts: Tuple[str, ...]) -> List[str]:
    paths: List[str] = []
    try:
        for fname in sorted(os.listdir(root)):
            if os.path.splitext(fname)[1].lower() in valid_exts:
                paths.append(os.path.join(root, fname))
    except FileNotFoundError:
        pass
    return paths


def load_image_bgr_u8(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f'No se pudo leer la imagen: {path}')
    return img


def mask_from_nonblack_u8(img_bgr_u8: np.ndarray, black_threshold: int = 0) -> np.ndarray:
    if black_threshold <= 0:
        non_black = (
            (img_bgr_u8[:, :, 0] > 0) |
            (img_bgr_u8[:, :, 1] > 0) |
            (img_bgr_u8[:, :, 2] > 0)
        )
    else:
        sum_bgr = (
            img_bgr_u8[:, :, 0].astype(np.int32) +
            img_bgr_u8[:, :, 1].astype(np.int32) +
            img_bgr_u8[:, :, 2].astype(np.int32)
        )
        non_black = (sum_bgr > black_threshold)
    return non_black


def to_float32_norm01(img_bgr_u8: np.ndarray) -> np.ndarray:
    return img_bgr_u8.astype(np.float32) / np.float32(255.0)


def apply_mask_mul(img_bgr_f32: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    return img_bgr_f32 * mask_bool.astype(np.float32)[..., None]


def to_rgb_float(img_bgr_f32: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr_f32, cv2.COLOR_BGR2RGB)


def to_lab_float(img_bgr_f32: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr_f32, cv2.COLOR_BGR2Lab)


def to_hsv_float(img_bgr_f32: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr_f32, cv2.COLOR_BGR2HSV)


def safe_mean_std(arr_f32: np.ndarray, mask_bool: np.ndarray):
    vals = arr_f32[mask_bool]
    if vals.size == 0:
        return float('nan'), float('nan')
    mean = float(vals.mean(dtype=np.float32))
    std = float(vals.std(ddof=1, dtype=np.float32)) if vals.size > 1 else float('nan')
    return mean, std


def compute_rgb_indices(img_rgb_f32: np.ndarray, mask_bool: np.ndarray) -> Dict[str, float]:
    R = img_rgb_f32[:, :, 0]
    G = img_rgb_f32[:, :, 1]
    B = img_rgb_f32[:, :, 2]

    ExG = (2.0 * G - R - B).astype(np.float32)
    VARI = ((G - R) / (G + R - B + EPS)).astype(np.float32)
    NGRDI = ((G - R) / (G + R + EPS)).astype(np.float32)
    G_over_R = (G / (R + EPS)).astype(np.float32)

    feats = {}
    for name, arr in (('ExG', ExG), ('VARI', VARI), ('NGRDI', NGRDI), ('G_over_R', G_over_R)):
        m, s = safe_mean_std(arr, mask_bool)
        feats[f'{name}_mean'] = m
        feats[f'{name}_std'] = s
    return feats


def compute_lab_features(img_lab_f32: np.ndarray, mask_bool: np.ndarray) -> Dict[str, float]:
    L = img_lab_f32[:, :, 0]
    a = img_lab_f32[:, :, 1]
    b = img_lab_f32[:, :, 2]

    L_mean, L_std = safe_mean_std(L, mask_bool)
    a_mean, a_std = safe_mean_std(a, mask_bool)
    b_mean, b_std = safe_mean_std(b, mask_bool)

    a_over_b = float(a_mean / (b_mean + float(EPS))) if not (np.isnan(a_mean) or np.isnan(b_mean)) else float('nan')

    colorfulness = np.sqrt(a * a + b * b, dtype=np.float32)
    colorfulness_mean, colorfulness_std = safe_mean_std(colorfulness, mask_bool)

    return {
        'L_mean': L_mean, 'L_std': L_std,
        'a_mean': a_mean, 'a_std': a_std,
        'b_mean': b_mean, 'b_std': b_std,
        'a_over_b': a_over_b,
        'colorfulness_mean': colorfulness_mean,
        'colorfulness_std': colorfulness_std,
    }


def circular_mean_std_deg(h_deg_vals: np.ndarray) -> Tuple[float, float]:
    if h_deg_vals.size == 0:
        return float('nan'), float('nan')
    ang = np.deg2rad(h_deg_vals).astype(np.float32)
    s = np.sin(ang).mean(dtype=np.float32)
    c = np.cos(ang).mean(dtype=np.float32)
    mean_ang = math.atan2(float(s), float(c))
    if mean_ang < 0:
        mean_ang += 2 * math.pi
    mean_deg = math.degrees(mean_ang)
    R = math.sqrt(float(s) ** 2 + float(c) ** 2)
    circ_std = math.sqrt(max(-2.0 * math.log(max(R, float(EPS))), 0.0))
    std_deg = math.degrees(circ_std)
    return float(mean_deg), float(std_deg)


def compute_hsv_features(img_hsv_f32: np.ndarray, mask_bool: np.ndarray) -> Dict[str, float]:
    H = img_hsv_f32[:, :, 0]
    S = img_hsv_f32[:, :, 1]
    V = img_hsv_f32[:, :, 2]

    H_vals = H[mask_bool]
    H_mean_deg, H_std_deg = circular_mean_std_deg(H_vals.astype(np.float32))

    S_mean, S_std = safe_mean_std(S, mask_bool)
    V_mean, V_std = safe_mean_std(V, mask_bool)

    return {
        'Hue_mean_deg': H_mean_deg, 'Hue_std_deg': H_std_deg,
        'S_mean': S_mean, 'S_std': S_std,
        'V_mean': V_mean, 'V_std': V_std,
    }


def extract_features_for_image(img_bgr_u8: np.ndarray, black_threshold: int) -> Dict[str, float]:
    mask_bool = mask_from_nonblack_u8(img_bgr_u8, black_threshold=black_threshold)
    img_bgr_f32 = to_float32_norm01(img_bgr_u8)
    masked_bgr_f32 = apply_mask_mul(img_bgr_f32, mask_bool)

    rgb = to_rgb_float(masked_bgr_f32)
    lab = to_lab_float(masked_bgr_f32)
    hsv = to_hsv_float(masked_bgr_f32)

    feats = {}
    feats.update(compute_rgb_indices(rgb, mask_bool))
    feats.update(compute_lab_features(lab, mask_bool))
    feats.update(compute_hsv_features(hsv, mask_bool))
    return feats


def run_pipeline(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)

    token_to_label: Dict[str, int] = {}
    for root in cfg.images_dirs:
        if not os.path.isdir(root):
            print(f'[AVISO] No existe el directorio: {root}', file=sys.stderr)
            continue
        token = infer_token_from_path(root)
        key = token.lower()
        if key in cfg.class_map_tokens:
            token_to_label[key] = cfg.class_map_tokens[key]
        else:
            num = auto_numeric_from_token(token)
            token_to_label[key] = num if num is not None else 0

    records = []
    for root in cfg.images_dirs:
        if not os.path.isdir(root):
            continue

        token = infer_token_from_path(root)
        key = token.lower()
        label = token_to_label.get(key, 0)

        if label == 0:
            print(f"[AVISO] No se pudo mapear token '{token}' a clase numérica. Ajusta CLASS_MAP_TOKENS.", file=sys.stderr)
            continue

        image_paths = list_images_in_root(root, cfg.valid_exts)
        if not image_paths:
            print(f'[AVISO] No se encontraron imágenes en: {root}', file=sys.stderr)
            continue

        for img_path in image_paths:
            try:
                img_bgr_u8 = load_image_bgr_u8(img_path)
                feats = extract_features_for_image(img_bgr_u8, cfg.black_threshold)

                row = {
                    'filename': os.path.basename(img_path),
                    'class_token': token,
                    'label': int(label)
                }
                row.update(feats)
                records.append(row)

            except Exception as e:
                print(f'[ERROR] {img_path}: {e}', file=sys.stderr)

    features_df = pd.DataFrame.from_records(records)
    if features_df.empty:
        raise RuntimeError('No se extrajeron características. Verifica rutas y formatos de imagen.')

    feats_csv = os.path.join(cfg.output_dir, 'caracteristicas_mango.csv')
    features_df.to_csv(feats_csv, index=False)
    print(f'[OK] Guardado DataFrame de características: {feats_csv}')


if __name__ == '__main__':
    cfg = Config(
        images_dirs=IMAGES_DIRS,
        output_dir=OUTPUT_DIR,
        class_map_tokens=parse_class_map_tokens(CLASS_MAP_TOKENS),
        black_threshold=BLACK_THRESHOLD,
    )
    run_pipeline(cfg)