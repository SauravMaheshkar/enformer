from typing import Dict

import tensorflow as tf

from .layers.activation_layers import GELU, SoftPlus
from .pooling import pooling_module
from .utils import TargetLengthCrop1D, exponential_linspace_int
from .layers.container_layers import Residual

SEQUENCE_LENGTH = 196_608
BIN_SIZE = 128
TARGET_LENGTH = 896


class Enformer(tf.keras.Model):
    def __init__(
        self,
        seed: int,
        channels: int = 1536,
        num_transformer_layers: int = 11,
        num_heads: int = 8,
        pooling_type: str = "max",
        name: str = "enformer",
        **kwargs,
    ) -> None:

        super(Enformer, self).__init__(name=name, **kwargs)

        heads_channels = {"human": 5313, "mouse": 1643}
        dropout_rate = 0.4
        assert channels % num_heads == 0, (
            "channels needs to be divisible " f"by {num_heads}"
        )

        trunk_name_scope = tf.name_scope("trunk")
        trunk_name_scope.__enter__()

        def conv_block(filters, width=1, w_init=None, name="conv_block", **kwargs):
            return tf.keras.Sequential(
                [
                    tf.keras.layers.BatchNormalization(
                        scale=True,
                        center=True,
                        momentum=0.9,
                        gamma_initializer=tf.keras.initializers.Ones(),
                    ),
                    GELU(),
                    tf.keras.layers.Conv1D(
                        filters=filters,
                        kernel_size=width,
                        kernel_initializer=w_init,
                        **kwargs,
                    ),
                ],
                name=name,
            )

        stem = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(filters=channels // 2, kernel_size=15),
                Residual(conv_block(channels // 2, 1, name="pointwise_conv_block")),
                pooling_module(pooling_type, pool_size=2),
            ],
            name = "stem"
        )

        filter_list = exponential_linspace_int(
            start=channels // 2, end=channels, num=6, divisible_by=128
        )
        conv_tower = tf.keras.Sequential(
            [
                tf.keras.Sequential(
                    [
                        conv_block(num_filters, 5),
                        Residual(
                            conv_block(num_filters, 1, name="pointwise_conv_block")
                        ),
                        pooling_module(pooling_type, pool_size=2),
                    ],
                    name=f"conv_tower_block_{i}",
                )
                for i, num_filters in enumerate(filter_list)
            ],
            name="conv_tower",
        )

        transformer = tf.keras.Sequential(
            [
                tf.keras.Sequential(
                    [
                        Residual(
                            tf.keras.Sequential(
                                [
                                    tf.keras.layers.LayerNormalization(
                                        axis=-1,
                                        scale=True,
                                        center=True,
                                        gamma_initializer=tf.keras.initializers.Ones(),
                                    ),
                                    tf.keras.layers.MultiHeadAttention(
                                        num_heads=num_heads,
                                        key_dim=64,
                                        value_dim=channels // num_heads,
                                        name=f"attention_{i}",
                                    ),
                                    tf.keras.layers.Dropout(
                                        rate=dropout_rate, seed=seed
                                    ),
                                ],
                                name = f"Attention_Block_{i}"
                            )
                        ),
                        Residual(
                            tf.keras.Sequential(
                                [
                                    tf.keras.layers.LayerNormalization(
                                        axis=-1, scale=True, center=True
                                    ),
                                    tf.keras.layers.Dense(units=channels * 2),
                                    tf.keras.layers.Dropout(
                                        rate=dropout_rate, seed=seed
                                    ),
                                    tf.keras.layers.ReLU(),
                                    tf.keras.layers.Dense(units=channels),
                                    tf.keras.layers.Dropout(
                                        rate=dropout_rate, seed=seed
                                    ),
                                ],
                                name = f"Attention_Head_{i}"
                            )
                        ),
                    ]
                )
                for i in range(num_transformer_layers)
            ],
            name="transformer",
        )

        crop_final = TargetLengthCrop1D(TARGET_LENGTH, name="target_input")

        final_pointwise = tf.keras.Sequential(
            [
                conv_block(channels * 2, 1),
                tf.keras.layers.Dropout(rate=dropout_rate / 8, seed=seed),
                GELU(),
            ],
            name="final_pointwise",
        )

        self._trunk = tf.keras.Sequential(
            [stem, conv_tower, 
            transformer, 
            crop_final, final_pointwise], name="trunk"
        )
        trunk_name_scope.__exit__(None, None, None)

        with tf.name_scope("heads"):
            self._heads = {
                head: tf.keras.Sequential(
                    [tf.keras.layers.Dense(units=num_channels), SoftPlus()],
                    name=f"head_{head}",
                )
                for head, num_channels in heads_channels.items()
            }

    @property
    def trunk(self):
        return self._trunk

    @property
    def heads(self):
        return self._heads

    def __call__(self, inputs: tf.Tensor) -> Dict[str, tf.Tensor]:
        trunk_embedding = self.trunk(inputs)
        return {
            head: head_module(trunk_embedding)
            for head, head_module in self.heads.items()
        }
