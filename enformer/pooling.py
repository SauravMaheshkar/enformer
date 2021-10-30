import tensorflow as tf

__all__ = ["pooling_module"]


def pooling_module(kind, pool_size):
    """Pooling module wrapper."""
    if kind == "attention":
        return SoftmaxPooling1D(pool_size=pool_size, per_channel=True, w_init_scale=2.0)
    elif kind == "max":
        return tf.keras.layers.MaxPool1D(pool_size=pool_size, padding="same")
    else:
        raise ValueError(f"Invalid pooling kind: {kind}.")


class SoftmaxPooling1D(tf.Module):
    """Pooling operation with optional weights."""

    def __init__(
        self,
        pool_size: int = 2,
        per_channel: bool = False,
        w_init_scale: float = 0.0,
        name: str = "softmax_pooling",
        **kwargs
    ):
        super(SoftmaxPooling1D, self).__init__(name=name, **kwargs)
        self._pool_size = pool_size
        self._per_channel = per_channel
        self._w_init_scale = w_init_scale
        self._logit_linear = None

    def _initialize(self, num_features):
        self._logit_linear = tf.keras.layers.Dense(
            units=num_features if self._per_channel else 1,
            use_bias=False,  # Softmax is agnostic to shifts.
            kernel_initializer=tf.keras.initializers.Identity(gain=self._w_init_scale),
        )

    @tf.Module.with_name_scope
    def __call__(self, inputs):
        _, length, num_features = inputs.shape
        self._initialize(num_features)
        inputs = tf.reshape(
            inputs, (-1, length // self._pool_size, self._pool_size, num_features)
        )
        return tf.reduce_sum(
            inputs * tf.nn.softmax(self._logit_linear(inputs), axis=-2), axis=-2
        )
