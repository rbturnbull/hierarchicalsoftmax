import pytest
from hierarchicalsoftmax import SoftmaxNode, HierarchicalSoftmaxLinear, HierarchicalSoftmaxLazyLinear
from hierarchicalsoftmax.layers import HierarchicalSoftmaxLayerError

def test_linear_layer():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    layer = HierarchicalSoftmaxLinear(in_features=100, root=root)

    assert layer.in_features == 100
    assert layer.out_features == 6

def test_lazy_linear_layer():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    layer = HierarchicalSoftmaxLazyLinear(root=root)

    assert layer.out_features == 6    


def test_no_explicit_out_features():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    with pytest.raises(HierarchicalSoftmaxLayerError):
        layer = HierarchicalSoftmaxLinear(in_features=100, out_features=100, root=root)

