from typing import Dict

import tensorflow as tf

__all__ = ["SoftmaxPooling1D"]


class SoftmaxPooling1D(tf.keras.layers.Layer):
    """Pooling operation with optional weights."""

    def __init__(
        self,
        pool_size: int = 2,
        per_channel: bool = False,
        w_init_scale: float = 0.0,
        name: str = "softmax_pooling",
        **kwargs,
    ) -> None:
        super(SoftmaxPooling1D, self).__init__(name=name, **kwargs)
        self._pool_size = pool_size
        self._per_channel = per_channel
        self._w_init_scale = w_init_scale

    def _initialize(self, num_features: int) -> None:
        self._logit_linear = tf.keras.layers.Dense(
            units=num_features if self._per_channel else 1,
            use_bias=False,  # Softmax is agnostic to shifts.
            kernel_initializer=tf.keras.initializers.Identity(gain=self._w_init_scale),
        )

    def get_config(self) -> Dict:
        config = {
            "pool_size": self._pool_size,
            "per_channel": self._per_channel,
            "w_init_scale": self._w_init_scale,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        _, length, num_features = inputs.shape
        self._initialize(num_features)
        inputs = tf.reshape(
            inputs, (-1, length // self._pool_size, self._pool_size, num_features)
        )
        return tf.reduce_sum(
            inputs * tf.nn.softmax(self._logit_linear(inputs), axis=-2), axis=-2
        )
