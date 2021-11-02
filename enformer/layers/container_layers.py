from typing import Dict
import tensorflow as tf

__all__ = ["Residual"]

class Residual(tf.keras.layers.Layer):
    def __init__(self, module: tf.Module, name="residual", **kwargs) -> None:
        super(Residual, self).__init__(name=name, **kwargs)
        self._module = module

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs + self._module(inputs)
    
    def get_config(self) -> Dict:
        config = {"module": self._module}
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape:tf.TensorShape) -> tf.TensorShape:
        return input_shape
