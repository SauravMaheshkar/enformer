import random
from typing import Any

import numpy as np

from enformer.nn import Enformer

TARGET_LENGTH = 896
SEQUENCE_LENGTH = 196_608


def test_enformer():
    model = Enformer(channels=1536, num_transformer_layers=11, seed=42)
    inputs = _get_random_input()
    outputs = model(inputs, training=True)
    assert outputs["human"].shape == (1, TARGET_LENGTH, 5313)
    assert outputs["mouse"].shape == (1, TARGET_LENGTH, 1643)


def one_hot_encode(
    sequence: str,
    alphabet: str = "ACGT",
    neutral_alphabet: str = "N",
    neutral_value: Any = 0,
    dtype=np.float32,
) -> np.ndarray:
    """One-hot encode sequence."""

    def to_uint8(string):
        return np.frombuffer(string.encode("ascii"), dtype=np.uint8)

    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8(neutral_alphabet)] = neutral_value
    hash_table = hash_table.astype(dtype)
    return hash_table[to_uint8(sequence)]


def _get_random_input():
    seq = "".join([random.choice("ACGT") for _ in range(SEQUENCE_LENGTH)])
    return np.expand_dims(one_hot_encode(seq), 0).astype(np.float32)
