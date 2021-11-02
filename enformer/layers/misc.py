from typing import Dict

import tensorflow as tf

__all__ = ["TargetLengthCrop1D"]


class TargetLengthCrop1D(tf.keras.layers.Layer):
    def __init__(self, target_length: int, name="target_length_crop", **kwargs) -> None:
        super(TargetLengthCrop1D, self).__init__(name=name, **kwargs)
        self._target_length = target_length

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        trim = (inputs.shape[-2] - self._target_length) // 2
        if trim < 0:
            raise ValueError("inputs longer than target length")

        return inputs[..., trim:-trim, :]

    def get_config(self) -> Dict:
        config = {"target_length": self._target_length}
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return input_shape
