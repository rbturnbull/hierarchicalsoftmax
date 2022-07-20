import torch
import pytest
from hierarchicalsoftmax.nodes import SoftmaxNode, IndexNotSetError
from hierarchicalsoftmax.inference import greedy_predictions, ShapeError

def test_greedy_predictions():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    root.set_indexes()

    targets = [aa,ba,bb, ab]

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    node_predictions = greedy_predictions(prediction_tensor=predictions, root=root)

    assert node_predictions == targets


def test_unset_indexes():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    targets = [aa,ba,bb, ab]

    predictions = torch.zeros( (len(targets), 10) )

    with pytest.raises(IndexNotSetError):
        node_predictions = greedy_predictions(prediction_tensor=predictions, root=root)


def test_greedy_predictions():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    root.set_indexes()

    targets = [aa,ba,bb, ab]

    predictions = torch.zeros( (len(targets), root.layer_size + 1) )
    with pytest.raises(ShapeError):
        node_predictions = greedy_predictions(prediction_tensor=predictions, root=root)
