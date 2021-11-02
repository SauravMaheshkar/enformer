import tensorflow as tf

__all__ = ["pooling_module"]

from .layers.pooling_layers import SoftmaxPooling1D


def pooling_module(kind, pool_size):
    """Pooling module wrapper."""
    if kind == "attention":
        return SoftmaxPooling1D(pool_size=pool_size, per_channel=True, w_init_scale=2.0)
    elif kind == "max":
        return tf.keras.layers.MaxPool1D(pool_size=pool_size, padding="same")
    else:
        raise ValueError(f"Invalid pooling kind: {kind}.")
