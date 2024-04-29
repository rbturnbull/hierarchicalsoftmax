import pytest
from torch import nn
from hierarchicalsoftmax import HierarchicalSoftmaxLinear, HierarchicalSoftmaxLazyLinear
from hierarchicalsoftmax.layers import HierarchicalSoftmaxLayerError
from hierarchicalsoftmax.tensors import LazyLinearTensor

import torch

from .util import depth_two_tree, assert_multiline_strings


def test_linear_layer():
    layer = HierarchicalSoftmaxLinear(in_features=100, root=depth_two_tree())

    assert layer.in_features == 100
    assert layer.out_features == 6

def test_lazy_linear_layer():
    layer = HierarchicalSoftmaxLazyLinear(root=depth_two_tree())

    assert layer.out_features == 6    


def test_no_explicit_out_features():
    with pytest.raises(HierarchicalSoftmaxLayerError):
        HierarchicalSoftmaxLinear(in_features=100, out_features=100, root=depth_two_tree())


def test_linear_model():
    model = nn.Sequential(
        nn.Linear(in_features=20, out_features=100),
        nn.ReLU(),
        HierarchicalSoftmaxLinear(in_features=100, root=depth_two_tree())
    )
    print(model)

    assert_multiline_strings( model, """
        Sequential(
            (0): Linear(in_features=20, out_features=100, bias=True)
            (1): ReLU()
            (2): HierarchicalSoftmaxLinear(in_features=100, out_features=6, bias=True)
        )
    """)


def test_forward():
    layer = HierarchicalSoftmaxLinear(in_features=100, root=depth_two_tree())
    x = torch.rand(10, 100)
    result = layer(x)
    assert result.shape == (10, 6)
    assert isinstance(result, LazyLinearTensor)
    assert result.result.shape == (10, 6)
    