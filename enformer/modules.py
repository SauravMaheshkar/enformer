from typing import Any, Dict, List, Optional

import tensorflow as tf

from .positional_features import positional_features_all
from .utils import relative_shift

__all__ = ["TransformerBlock"]


class TransformerBlock(tf.Module):
    """Full transformer module block."""

    def __init__(
        self,
        channels: int,
        dropout_rate: float,
        attention_kwargs: Dict[str, Any],
        seed: int = 42,
        name: str = "transformer_block",
        **kwargs
    ):
        super(TransformerBlock, self).__init__(name=name, **kwargs)

        # ================ MultiHeadAttention ===================

        self.mha_ln = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, scale=True, center=True, name=name + "_mha_layernorm"
        )
        self.mha = MultiheadAttention(**attention_kwargs)
        self.mha_dropout = tf.keras.layers.Dropout(
            rate=dropout_rate, seed=seed, name=name + "_mha_dropout"
        )

        # ================ MultiLayerPerceptron ===================

        initializer: Any = tf.keras.initializers.LecunNormal(seed=seed)

        self.mlp_ln = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, scale=True, center=True, name=name + "_mlp_layernorm"
        )
        self.mlp_linear1 = tf.keras.layers.Dense(
            units=channels * 2,
            kernel_initializer=initializer,
            name=name + "_mlp_linear_1",
        )
        self.mlp_dropout1 = tf.keras.layers.Dropout(
            rate=dropout_rate, seed=seed, name=name + "_mlp_dropout_1"
        )
        self.mlp_linear2 = tf.keras.layers.Dense(
            units=channels, kernel_initializer=initializer, name=name + "_mlp_linear_2"
        )
        self.mlp_dropout2 = tf.keras.layers.Dropout(
            rate=dropout_rate, seed=seed, name=name + "_mlp_dropout_2"
        )

    @tf.Module.with_name_scope
    def __call__(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:

        # ================ MultiHeadAttention ===================

        x = self.mha_ln(inputs)
        x = self.mha(x, training=training)
        x = self.mha_dropout(x, training=training)
        x += inputs  # Residual Connection
        mha_output = x

        # ================ MultiLayerPerceptron ===================

        x = self.mlp_ln(mha_output)
        x = self.mlp_linear1(x)
        x = self.mlp_dropout1(x, training=training)
        x = tf.nn.relu(x)
        x = self.mlp_linear2(x)
        x = self.mlp_dropout2(x, training=training)
        return x + mha_output  # Residual Connection


class MultiheadAttention(tf.Module):
    """Multi-head attention."""

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
    ):

        super(MultiheadAttention, self).__init__(name=name, **kwargs)
        self._value_size = value_size
        self._key_size = key_size
        self._num_heads = num_heads
        self._attention_dropout_rate = attention_dropout_rate
        self._scaling = scaling
        self._relative_positions = relative_positions
        self._relative_position_symmetric = relative_position_symmetric
        self._relative_position_functions = relative_position_functions
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

        key_proj_size = self._key_size * self._num_heads
        embedding_size = self._value_size * self._num_heads

        # ================ Q,K,V Layers ===================

        self._q_layer = tf.keras.layers.Dense(
            units=key_proj_size,
            name=name + "_q_layer",
            use_bias=False,
            kernel_initializer=self._initializer,
        )
        self._k_layer = tf.keras.layers.Dense(
            units=key_proj_size,
            name=name + "_k_layer",
            use_bias=False,
            kernel_initializer=self._initializer,
        )
        self._v_layer = tf.keras.layers.Dense(
            units=embedding_size,
            name=name + "_v_layer",
            use_bias=False,
            kernel_initializer=self._initializer,
        )

        w_init = tf.keras.initializers.Zeros() if zero_initialize else self._initializer

        self._embedding_layer = tf.keras.layers.Dense(
            units=embedding_size,
            name=name + "_embedding_layer",
            kernel_initializer=w_init,
        )

        # ================ Additional Layers for Relative Positions ===================
        if self._relative_positions:
            self._r_k_layer = tf.keras.layers.Dense(
                units=key_proj_size,
                name=name + "_r_k_layer",
                use_bias=False,
                kernel_initializer=self._initializer,
            )
            self._r_w_bias = tf.Variable(
                self._initializer(
                    [1, self._num_heads, 1, self._key_size], dtype=tf.float32
                ),
                name=name + "_r_w_bias",
            )
            self._r_r_bias = tf.Variable(
                self._initializer(
                    [1, self._num_heads, 1, self._key_size], dtype=tf.float32
                ),
                name=name + "_r_r_bias",
            )

    def _multihead_output(self, linear, inputs):
        """Applies a standard linear to inputs and returns multihead output."""

        output = linear(inputs) # [B, T, H * KV]
        num_kv_channels = output.shape[-1] // self._num_heads
        # Split H * Channels into separate axes.
        output = tf.reshape(tensor=output, shape=[-1, self._num_heads, num_kv_channels])
        # [B, T, H, KV] -> [B, H, T, KV]
        return tf.transpose(output, perm = [0, 2, 1, 3])

    @tf.Module.with_name_scope
    def __call__(self, inputs, training: bool = False):

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
