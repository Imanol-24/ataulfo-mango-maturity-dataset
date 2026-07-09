import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.initializers import HeNormal, VarianceScaling
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.regularizers import l2


def act_layer(name):
    name = (name or "leakyrelu").lower()
    if name == "softsign":
        return tf.keras.layers.Activation("softsign")
    return tf.keras.layers.LeakyReLU(negative_slope=0.1)


def dense_block(units, act="leakyrelu", use_bn=True, dropout=0.0, l2_reg=0.0, init_name="variancescaling"):
    kernel_init = VarianceScaling() if init_name == "variancescaling" else HeNormal()

    layers = [
        Dense(
            units,
            use_bias=not use_bn,
            kernel_initializer=kernel_init,
            kernel_regularizer=l2(l2_reg) if l2_reg and l2_reg > 0 else None,
        )
    ]

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
    use_bn=True,
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


def make_optimizer(lr=3e-3):
    return tf.keras.optimizers.Adam(learning_rate=lr)


def make_callbacks(checkpoint_path):
    return [
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=0),
        EarlyStopping(monitor="val_loss", patience=18, restore_best_weights=True, verbose=0),
    ]