import numpy as np
import tensorflow as tf

__all__ = ["relative_shift"]


def relative_shift(x):
    """Shift the relative logits like in TransformerXL."""
    to_pad = tf.zeros_like(x[..., :1])
    x = tf.concat([to_pad, x], -1)
    _, num_heads, t1, t2 = x.shape
    x = tf.reshape(x, [-1, num_heads, t2, t1])
    x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, [-1, num_heads, t1, t2 - 1])
    x = tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, (t2 + 1) // 2])
    return x


def exponential_linspace_int(start, end, num, divisible_by=1):
    def _round(x):
        return int(np.round(x / divisible_by) * divisible_by)

    base = np.exp(np.log(end / start) / (num - 1))
    return [_round(start * base ** i) for i in range(num)]
class TargetLengthCrop1D(tf.Module):
    def __init__(self, target_length: int, name="target_length_crop", **kwargs):
        super(TargetLengthCrop1D, self).__init__(name=name, **kwargs)
        self._target_length = target_length

    @tf.Module.with_name_scope
    def __call__(self, inputs):
        trim = (inputs.shape[-2] - self._target_length) // 2
        if trim < 0:
            raise ValueError("inputs longer than target length")

        return inputs[..., trim:-trim, :]
