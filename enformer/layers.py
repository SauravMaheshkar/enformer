from typing import Dict, Optional

import tensorflow as tf

__all__ = ["SoftPlus", "GELU"]


# ================ SoftPlus ===================


class SoftPlus(tf.keras.layers.Layer):
    def __init__(self, name: Optional[str] = "softplus", **kwargs):
        super(SoftPlus, self).__init__(name=name, **kwargs)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        return tf.math.softplus(features=input, name=self.name)

    def compute_output_shape(self, input_shape):
        return input_shape


# ================ GELU ===================


class GELU(tf.keras.layers.Layer):
    def __init__(
        self, approximate: bool = True, name: Optional[str] = "gelu", **kwargs
    ):
        super(GELU, self).__init__(name=name, **kwargs)
        self.approximate = approximate
        self.supports_masking = True

    def call(self, input) -> tf.Tensor:
        return gelu(input, approximate=self.approximate)

    def get_config(self) -> Dict:
        config = {"approximate": self.approximate}
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape


def gelu(x: tf.Tensor, approximate: bool = True) -> tf.Tensor:
    return tf.nn.gelu(x, approximate)
