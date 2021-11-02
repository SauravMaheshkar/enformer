from typing import Dict

import tensorflow as tf

__all__ = ["Identity"]


class Identity(tf.keras.layers.Layer):
    def __init__(self, name="identity", **kwargs) -> None:
        super(Identity, self).__init__(name=name, **kwargs)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs

    def get_config(self) -> Dict:
        base_config = super().get_config()
        return base_config

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return input_shape
