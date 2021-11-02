import pytest
import tensorflow as tf

from enformer.layers.activation_layers import GELU, SoftPlus
from enformer.layers.container_layers import Residual
from enformer.layers.pooling_layers import SoftmaxPooling1D

from .utils import Identity


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


@pytest.mark.layers
@pytest.mark.actions
def test_residual():

    layer = Residual(Identity(), name="residual")
    assert layer.name == "residual"

    outputs = layer(inputs=tf.ones((2, 2)))
    assert outputs.shape == tf.TensorShape((2, 2))


@pytest.mark.layers
@pytest.mark.actions
def test_softmax_pooling():

    layer = SoftmaxPooling1D()
    assert layer.name == "softmax_pooling"

    outputs = layer((tf.ones([3, 4, 5])))
    assert outputs.shape == tf.TensorShape((3, 2, 5))
