import pytest
import tensorflow as tf

from enformer.layers import GELU, SoftPlus

@pytest.mark.layers
@pytest.mark.actions
def test_softplus():

    layer = SoftPlus("softplus")
    assert layer.name == "softplus"

    outputs = layer(tf.range(0, 2, dtype=tf.float32)).numpy()
    assert len(outputs) == 2


@pytest.mark.layers
@pytest.mark.actions
def test_gelu():

    layer = GELU("gelu")
    assert layer.name == "gelu"

    outputs = layer(tf.range(0, 2, dtype=tf.float32)).numpy()
    assert len(outputs) == 2
