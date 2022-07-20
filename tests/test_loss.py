import pytest
from hierarchicalsoftmax.loss import HierarchicalSoftmaxLoss
from hierarchicalsoftmax.nodes import SoftmaxNode, ReadOnlyError

import torch

def test_loss():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    loss = HierarchicalSoftmaxLoss(root)

    targets = [aa,ba,bb, ab]
    target_tensor = root.get_node_ids_tensor(targets)

    predictions = torch.zeros( (len(targets), root.layer_size) )

    # Test blank is inaccurate
    value = loss(predictions, target_tensor)
    assert value > 1.38

    # Test accurate
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    value = loss(predictions, target_tensor)
    assert value < 0.0001


def test_read_only():
    """ Ensures that you cannot add nodes after building the loss. """
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)

    loss = HierarchicalSoftmaxLoss(root)

    with pytest.raises(ReadOnlyError):
        b = SoftmaxNode("b", parent=root)

