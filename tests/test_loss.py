import pytest
from hierarchicalsoftmax.loss import HierarchicalSoftmaxLoss, focal_loss_with_smoothing
from hierarchicalsoftmax.nodes import SoftmaxNode, ReadOnlyError
import torch.nn.functional as F
from torch.testing import assert_close

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

    assert root.layer_size == 6

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


def test_loss_with_single_child():
    root = SoftmaxNode("root")
    a_parent = SoftmaxNode("a_parent", parent=root)
    a = SoftmaxNode("a", parent=a_parent)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    loss = HierarchicalSoftmaxLoss(root)

    targets = [aa,ba,bb, ab]
    target_tensor = root.get_node_ids_tensor(targets)

    predictions = torch.zeros( (len(targets), root.layer_size) )

    assert root.layer_size == 6

    # Test blank is inaccurate
    value = loss(predictions, target_tensor)
    assert value > 1.38

    # Test accurate
    for target_index, target in enumerate(targets):
        while target.parent:
            # If the parent has only one child, then skip it
            if len(target.parent.children) == 1:
                target = target.parent
                continue
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


def test_focal_loss():
    gamma = 2.0
    root = SoftmaxNode("root", gamma=gamma)
    a = SoftmaxNode("a", parent=root, gamma=gamma)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root, gamma=gamma)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    loss = HierarchicalSoftmaxLoss(root)

    targets = [aa,ba,bb, ab]
    target_tensor = root.get_node_ids_tensor(targets)

    predictions = torch.zeros( (len(targets), root.layer_size) )

    # Test blank is inaccurate but not as bad as cross entropy
    value = loss(predictions, target_tensor)
    assert 0.3 < value < 1.0

    # Test accurate
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    value = loss(predictions, target_tensor)
    assert value < 0.0001


def test_focal_loss_with_smoothing():
    target = torch.as_tensor([0], dtype=int).long()
    logits = torch.as_tensor([[0.0, 1.0]])
    
    assert_close(
        F.cross_entropy(logits, target), 
        focal_loss_with_smoothing(logits, target)
    )

    label_smoothing = 0.1
    assert_close(
        F.cross_entropy(logits, target, label_smoothing=label_smoothing), 
        focal_loss_with_smoothing(logits, target, label_smoothing=label_smoothing)
    )

    label_smoothing = 0.2
    assert_close(
        F.cross_entropy(logits, target, label_smoothing=label_smoothing), 
        focal_loss_with_smoothing(logits, target, label_smoothing=label_smoothing)
    )

    gamma = 1.0
    assert_close( 
        (1.0 - 0.2689414322376251) ** gamma * F.cross_entropy(logits, target), 
        focal_loss_with_smoothing(logits, target, gamma=gamma) 
    )

    gamma = 2.0
    assert_close( 
        (1.0 - 0.2689414322376251) ** gamma * F.cross_entropy(logits, target), 
        focal_loss_with_smoothing(logits, target, gamma=gamma) 
    )

    assert focal_loss_with_smoothing(logits, target, gamma=gamma) > focal_loss_with_smoothing(logits, target, gamma=gamma, label_smoothing=0.1)

    assert focal_loss_with_smoothing(logits, target, gamma=gamma, label_smoothing=0.2) < focal_loss_with_smoothing(logits, target, gamma=gamma, label_smoothing=0.1)


def test_focal_loss_with_smoothing_and_weights():
    target = torch.as_tensor([0, 1], dtype=int).long()
    logits = torch.as_tensor([[0.0, 1.0], [0.0, 0.5]])
    
    assert_close(
        F.cross_entropy(logits, target),
        focal_loss_with_smoothing(logits, target)
    )

    weight = torch.as_tensor([1.0, 1.0], dtype=torch.float32)
    assert_close(
        F.cross_entropy(logits, target), # 0.8937
        focal_loss_with_smoothing(logits, target, weight=weight)
    )
    assert_close(
        F.cross_entropy(logits, target, weight=weight), # 0.8937
        focal_loss_with_smoothing(logits, target, weight=weight)
    )

    weight = torch.as_tensor([0.5, 1.0], dtype=torch.float32)
    assert_close(
        F.cross_entropy(logits, target, weight=weight),
        focal_loss_with_smoothing(logits, target, weight=weight)
    )

    weight = torch.as_tensor([1.35, 0.9], dtype=torch.float32)
    assert_close(
        F.cross_entropy(logits, target, weight=weight),
        focal_loss_with_smoothing(logits, target, weight=weight)
    )
