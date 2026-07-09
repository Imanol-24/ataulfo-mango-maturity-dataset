import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def set_seed(seed=1337):
    import tensorflow as tf
    np.random.seed(seed)
    tf.random.set_seed(seed)


def read_numeric_csv(path):
    df = pd.read_csv(path)
    df = df.select_dtypes(include=[np.number]).copy()
    df = df.fillna(0)
    return df


def load_four_csvs(file_r01, file_r02, file_r03, file_r04):
    d1 = read_numeric_csv(file_r01)
    d2 = read_numeric_csv(file_r02)
    d3 = read_numeric_csv(file_r03)
    d4 = read_numeric_csv(file_r04)

    common_cols = sorted(list(set(d1.columns) & set(d2.columns) & set(d3.columns) & set(d4.columns)))
    if not common_cols:
        raise ValueError("No hay columnas numéricas comunes entre los CSV.")

    x0 = d1[common_cols].values
    x1 = d2[common_cols].values
    x2 = d3[common_cols].values
    x3 = d4[common_cols].values

    y0 = np.zeros((len(x0),), dtype=np.int32)
    y1 = np.ones((len(x1),), dtype=np.int32)
    y2 = np.full((len(x2),), 2, dtype=np.int32)
    y3 = np.full((len(x3),), 3, dtype=np.int32)

    x = np.vstack([x0, x1, x2, x3]).astype("float32")
    y = np.concatenate([y0, y1, y2, y3]).astype("int32")
    return x, y, common_cols


def split_and_scale_data(x, y, val_size, test_size, seed):
    x_train, x_rest, y_train, y_rest = train_test_split(
        x,
        y,
        test_size=(val_size + test_size),
        stratify=y,
        random_state=seed,
    )

    rel_test = test_size / (val_size + test_size)

    x_val, x_test, y_val, y_test = train_test_split(
        x_rest,
        y_rest,
        test_size=rel_test,
        stratify=y_rest,
        random_state=seed,
    )

    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return x_train, x_val, x_test, y_train, y_val, y_test, scaler


def class_weight_to_sample_weight(y, cw_map):
    return np.asarray([cw_map[int(c)] for c in y], dtype="float32")


def rebuild_split_indices(y, val_size, test_size, seed):
    all_idx = np.arange(len(y))

    idx_train, idx_rest = train_test_split(
        all_idx,
        test_size=(val_size + test_size),
        stratify=y,
        random_state=seed,
    )

    y_rest = y[idx_rest]
    rel_test = test_size / (val_size + test_size)

    idx_val, idx_test = train_test_split(
        idx_rest,
        test_size=rel_test,
        stratify=y_rest,
        random_state=seed,
    )

    return {
        "train": idx_train,
        "val": idx_val,
        "test": idx_test,
    }