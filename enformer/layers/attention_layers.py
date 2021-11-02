from typing import Callable, Dict, List, Optional

import sonnet as snt
import tensorflow as tf
from enformer.utils.positional import positional_features_all, relative_shift

__all__ = ["MultiheadAttention"]


class MultiheadAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        value_size: int,
        key_size: int,
        num_heads: int,
        scaling: bool = True,
        seed: int = 42,
        attention_dropout_rate: float = 0.1,
        relative_positions: bool = False,
        relative_position_symmetric: bool = False,
        relative_position_functions: Optional[List[str]] = None,
        num_relative_position_features: Optional[int] = None,
        positional_dropout_rate: float = 0.1,
        zero_initialize: bool = True,
        initializer: Optional[tf.keras.initializers.Initializer] = None,
        name: str = "multihead_attention",
        **kwargs
    ) -> None:

        super(MultiheadAttention, self).__init__(name=name, **kwargs)
        self._value_size = value_size
        self._key_size = key_size
        self._num_heads = num_heads
        self._attention_dropout_rate = attention_dropout_rate
        self._scaling = scaling
        self._relative_positions = relative_positions
        self._relative_position_symmetric = relative_position_symmetric
        self._relative_position_functions = relative_position_functions
        self.zero_initialize = zero_initialize
        self.key_proj_size = self._key_size * self._num_heads
        self.embedding_size = self._value_size * self._num_heads
        if num_relative_position_features is None:
            # num_relative_position_features needs to be divisible by the number of
            # relative positional functions *2 (for symmetric & asymmetric version).
            divisible_by = 2 * len(self._relative_position_functions)  # type:ignore
            self._num_relative_position_features = (
                self._value_size // divisible_by
            ) * divisible_by
        else:
            self._num_relative_position_features = num_relative_position_features
        self._positional_dropout_rate = positional_dropout_rate

        self._initializer = initializer
        if self._initializer is None:
            self._initializer = tf.keras.initializers.VarianceScaling(
                scale=2.0, seed=seed
            )

    def build(self, input_shape: tf.TensorShape) -> None:

        # ================ Q,K,V Layers ===================

        self._q_layer = tf.keras.layers.Dense(
            units=self.key_proj_size,
            name=self.name + "_q_layer",
            use_bias=False,
            kernel_initializer=self._initializer,
        )
        self._k_layer = tf.keras.layers.Dense(
            units=self.key_proj_size,
            name=self.name + "_k_layer",
            use_bias=False,
            kernel_initializer=self._initializer,
        )
        self._v_layer = tf.keras.layers.Dense(
            units=self.embedding_size,
            name=self.name + "_v_layer",
            use_bias=False,
            kernel_initializer=self._initializer,
        )

        w_init = (
            tf.keras.initializers.Zeros() if self.zero_initialize else self._initializer
        )

        self._embedding_layer = tf.keras.layers.Dense(
            units=self.embedding_size,
            name=self.name + "_embedding_layer",
            kernel_initializer=w_init,
        )

        # ================ Additional Layers for Relative Positions ===================
        if self._relative_positions:
            self._r_k_layer = tf.keras.layers.Dense(
                units=self.key_proj_size,
                name=self.name + "_r_k_layer",
                use_bias=False,
                kernel_initializer=self._initializer,
            )
            self._r_w_bias = tf.Variable(
                self._initializer(
                    [1, self._num_heads, 1, self._key_size], dtype=tf.float32
                ),
                name=self.name + "_r_w_bias",
            )
            self._r_r_bias = tf.Variable(
                self._initializer(
                    [1, self._num_heads, 1, self._key_size], dtype=tf.float32
                ),
                name=self.name + "_r_r_bias",
            )

    def _multihead_output(self, linear: Callable, inputs: tf.Tensor) -> tf.Tensor:
        """Applies a standard linear to inputs and returns multihead output."""

        output = snt.BatchApply(linear)(inputs)  # [B, T, H * KV]
        assert len(output) == 4, f"len of output is not 4, but rather {len(output)}"
        num_kv_channels = output.shape[-1] // self._num_heads
        # Split H * Channels into separate axes.
        output = tf.reshape(tensor=output, shape=[-1, self._num_heads, num_kv_channels])
        # [B, T, H, KV] -> [B, H, T, KV]
        assert len(output) == 4
        return tf.transpose(output, perm=[0, 2, 1, 3])

    def get_config(self) -> Dict:
        config = {
            "value_size": self._value_size,
            "key_size": self._key_size,
            "num_heads": self._num_heads,
            "attention_dropout_rate": self._attention_dropout_rate,
            "scaling": self._scaling,
            "relative_positions": self._relative_positions,
            "relative_position_symmetric": self._relative_position_symmetric,
            "relative_position_functions": self._relative_position_functions,
            "zero_initialize": self.zero_initialize,
            "key_proj_size": self.key_proj_size,
            "embedding_size": self.embedding_size,
            "num_relative_position_features": self._num_relative_position_features,
            "positional_dropout_rate": self._positional_dropout_rate,
            "initializer": self._initializer,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:

        # Initialise the projection layers.
        embedding_size = self._value_size * self._num_heads
        seq_len = inputs.shape[1]

        # ================ Q,K,V Layers ===================

        q = self._multihead_output(self._q_layer, inputs)  # [B, H, T, K]
        k = self._multihead_output(self._k_layer, inputs)  # [B, H, T, K]
        v = self._multihead_output(self._v_layer, inputs)  # [B, H, T, V]

        # Scale the query by the square-root of key size.
        if self._scaling:
            q *= self._key_size ** -0.5

        if self._relative_positions:
            # For relative positions, we project positions to form relative keys.
            distances = tf.range(-seq_len + 1, seq_len, dtype=tf.float32)[tf.newaxis]
            positional_encodings = positional_features_all(
                positions=distances,
                feature_size=self._num_relative_position_features,
                seq_length=seq_len,
                feature_functions=self._relative_position_functions,
                symmetric=self._relative_position_symmetric,
            )
            # [1, 2T-1, Cr]

        if training:
            positional_encodings = tf.nn.dropout(
                positional_encodings, rate=self._positional_dropout_rate
            )

            # [1, H, 2T-1, K]
            r_k = self._r_k_layer(positional_encodings)

            # Add shifted relative logits to content logits.
            # [B, H, T', T]
            content_logits = tf.matmul(q + self._r_w_bias, k, transpose_b=True)
            # [B, H, T', 2T-1]
            relative_logits = tf.matmul(q + self._r_r_bias, r_k, transpose_b=True)
            #  [B, H, T', T]
            relative_logits = relative_shift(relative_logits)
            logits = content_logits + relative_logits
        else:
            # [B, H, T', T]
            logits = tf.matmul(q, k, transpose_b=True)

            weights = tf.nn.softmax(logits)

        # Dropout on the attention weights.
        if training:
            weights = tf.nn.dropout(weights, rate=self._attention_dropout_rate)

            # Transpose and reshape the output.
            output = tf.matmul(weights, v)  # [B, H, T', V]
            output_transpose = tf.transpose(output, [0, 2, 1, 3])  # [B, T', H, V]

            # Final linear layer.
            attended_inputs = tf.reshape(
                tensor=output_transpose, shape=[embedding_size], preserve_dims=2
            )
            output = self._embedding_layer(attended_inputs)

        return output
