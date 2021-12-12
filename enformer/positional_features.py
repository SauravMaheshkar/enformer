from typing import List, Optional

import numpy as np
import tensorflow as tf

__all__ = ["positional_features_all"]


def get_positional_feature_function(name):
    available = {
        "positional_features_exponential": positional_features_exponential,
        "positional_features_central_mask": positional_features_central_mask,
        "positional_features_gamma": positional_features_gamma,
        "positional_features_cosine": positional_features_cosine,
        "positional_features_linear_masks": positional_features_linear_masks,
        "positional_features_sin_cos": positional_features_sin_cos,
    }
    if name not in available:
        raise ValueError(f"Function {name} not available in {available.keys()}")
    return available[name]


def positional_features_all(
    positions: tf.Tensor,
    feature_size: int,
    seq_length: Optional[int] = None,
    bin_size: Optional[int] = None,
    feature_functions: Optional[List[str]] = None,
    symmetric=False,
):
    if feature_functions is None:
        feature_functions = [
            "positional_features_exponential",
            "positional_features_central_mask",
            "positional_features_gamma",
        ]
    num_components = len(feature_functions)  # 1 per each basis function
    if not symmetric:
        num_components = 2 * num_components

    # For now, we do not allow odd sized embeddings.
    if feature_size % num_components != 0:
        raise ValueError(f"feature_size has to be divisible by {num_components}")

    feature_functions = [get_positional_feature_function(f) for f in feature_functions]
    num_basis_per_class = feature_size // num_components
    embeddings = tf.concat(
        [
            func(
                tf.abs(positions), num_basis_per_class, seq_length, bin_size
            )  # type:ignore
            for func in feature_functions
        ],
        axis=-1,
    )
    if not symmetric:
        embeddings = tf.concat(
            [embeddings, tf.sign(positions)[..., tf.newaxis] * embeddings], axis=-1
        )
    tf.TensorShape(embeddings.shape).assert_is_compatible_with(
        positions.shape + [feature_size]
    )
    return embeddings


def _prepend_dims(x, num_dims):
    return tf.reshape(x, shape=[1] * num_dims + x.shape)


def positional_features_exponential(
    positions: tf.Tensor,
    feature_size: int,
    seq_length: Optional[int] = None,
    bin_size: Optional[int] = None,
    min_half_life: Optional[float] = 3.0,
):

    del bin_size  # Unused.
    if seq_length is None:
        seq_length = tf.reduce_max(tf.abs(positions)) + 1
    # Grid of half lifes from [3, seq_length / 2] with feature_size
    # distributed on the log scale.
    seq_length = tf.cast(seq_length, dtype=tf.float32)
    max_range = tf.math.log(seq_length) / tf.math.log(2.0)
    half_life = tf.pow(2.0, tf.linspace(min_half_life, max_range, feature_size))
    half_life = _prepend_dims(half_life, positions.shape.rank)
    positions = tf.abs(positions)
    outputs = tf.exp(-tf.math.log(2.0) / half_life * positions[..., tf.newaxis])
    tf.TensorShape(outputs.shape).assert_is_compatible_with(
        positions.shape + [feature_size]
    )
    return outputs


def positional_features_central_mask(
    positions: tf.Tensor,
    feature_size: int,
    seq_length: Optional[int] = None,
    bin_size: Optional[int] = None,
):
    """Positional features using a central mask (allow only central features)."""
    del seq_length  # Unused.
    del bin_size  # Unused.
    center_widths = tf.pow(2.0, tf.range(1, feature_size + 1, dtype=tf.float32))
    center_widths = center_widths - 1
    center_widths = _prepend_dims(center_widths, positions.shape.rank)
    outputs = tf.cast(center_widths > tf.abs(positions)[..., tf.newaxis], tf.float32)
    tf.TensorShape(outputs.shape).assert_is_compatible_with(
        positions.shape + [feature_size]
    )
    return outputs


def gamma_pdf(x, concentration, rate):
    """Gamma probability distribution function: p(x|concentration, rate)."""
    log_unnormalized_prob = tf.math.xlogy(concentration - 1.0, x) - rate * x
    log_normalization = tf.math.lgamma(concentration) - concentration * tf.math.log(
        rate
    )
    return tf.exp(log_unnormalized_prob - log_normalization)


def positional_features_gamma(
    positions: tf.Tensor,
    feature_size: int,
    seq_length: Optional[int] = None,
    bin_size: Optional[int] = None,
    stddev=None,
    start_mean=None,
):
    """Positional features computed using the gamma distributions."""
    del bin_size  # Unused.
    if seq_length is None:
        seq_length = tf.reduce_max(tf.abs(positions)) + 1
    if stddev is None:
        stddev = seq_length / (2 * feature_size)
    if start_mean is None:
        start_mean = seq_length / feature_size
    mean = tf.linspace(start_mean, seq_length, num=feature_size)
    mean = _prepend_dims(mean, positions.shape.rank)
    concentration = (mean / stddev) ** 2
    rate = mean / stddev ** 2
    probabilities = gamma_pdf(
        tf.abs(tf.cast(positions, dtype=tf.float32))[..., tf.newaxis],
        concentration,
        rate,
    )
    probabilities += 1e-8  # To ensure numerical stability.
    outputs = probabilities / tf.reduce_max(probabilities)
    tf.TensorShape(outputs.shape).assert_is_compatible_with(
        positions.shape + [feature_size]
    )
    return outputs


def positional_features_cosine(
    positions: tf.Tensor,
    feature_size: int,
    seq_length: Optional[int] = None,
    bin_size: Optional[int] = None,
):
    """Cosine positional features."""
    del bin_size  # Unused.
    del seq_length  # Unused.
    periodicity = 1.25 * tf.pow(2.0, tf.range(0, feature_size, dtype=tf.float32))
    periodicity = _prepend_dims(periodicity, positions.shape.rank)

    outputs = tf.math.cos(2 * np.pi * positions[..., tf.newaxis] / periodicity)
    tf.TensorShape(outputs.shape).assert_is_compatible_with(
        positions.shape + [feature_size]
    )
    return outputs


def positional_features_linear_masks(
    positions: tf.Tensor,
    feature_size: int,
    seq_length: Optional[int] = None,
    bin_size: Optional[int] = None,
):
    """Exponentially increasing point focuses."""
    del bin_size  # Unused.
    del seq_length  # Unused.
    distances = tf.range(0, feature_size, dtype=tf.float32)
    distances = _prepend_dims(distances, positions.shape.rank)
    outputs = tf.cast(distances == tf.abs(positions[..., tf.newaxis]), dtype=tf.float32)

    tf.TensorShape(outputs.shape).assert_is_compatible_with(
        positions.shape + [feature_size]
    )
    return outputs


def positional_features_sin_cos(
    positions: tf.Tensor,
    feature_size: int,
    seq_length: Optional[int] = None,
    bin_size: Optional[int] = None,
    max_time=10000.0,
):
    """Sine/cosine positional encodings."""
    del bin_size  # Unused.
    del seq_length  # Unused.
    if feature_size % 2 != 0:
        raise ValueError("feature_size needs to be divisible by 2.")
    i = tf.range(0, feature_size, 2, dtype=tf.float32)
    i = _prepend_dims(i, positions.shape.rank)

    # Concat sines and cosines and return.
    outputs = tf.concat(
        [
            tf.sin(positions[..., tf.newaxis] / max_time ** (i / feature_size)),
            tf.cos(positions[..., tf.newaxis] / max_time ** (i / feature_size)),
        ],
        -1,
    )

    tf.TensorShape(outputs.shape).assert_is_compatible_with(
        positions.shape + [feature_size]
    )
    return outputs
