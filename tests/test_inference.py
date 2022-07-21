import torch
import pytest
from hierarchicalsoftmax.nodes import SoftmaxNode, IndexNotSetError
from hierarchicalsoftmax.inference import greedy_predictions, ShapeError

def get_root_and_targets():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    targets = [aa,ba,bb, ab]

    return root, targets

def test_greedy_predictions():
    root, targets = get_root_and_targets()
    root.set_indexes()

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    node_predictions = greedy_predictions(prediction_tensor=predictions, root=root)

    assert node_predictions == targets


def test_unset_indexes():
    root, targets = get_root_and_targets()

    predictions = torch.zeros( (len(targets), 10) )

    with pytest.raises(IndexNotSetError):
        greedy_predictions(prediction_tensor=predictions, root=root)


def test_greedy_predictions():
    root, targets = get_root_and_targets()

    root.set_indexes()

    predictions = torch.zeros( (len(targets), root.layer_size + 1) )
    with pytest.raises(ShapeError):
        node_predictions = greedy_predictions(prediction_tensor=predictions, root=root)


def test_max_depth_simple():
    root, targets = get_root_and_targets()

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent
    node_predictions = greedy_predictions(prediction_tensor=predictions, root=root, max_depth=1)
    assert [str(node) for node in node_predictions] == ['a', 'b', 'b', 'a']

def test_max_depth_complex():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    aaa = SoftmaxNode("aaa", parent=aa)
    aab = SoftmaxNode("aab", parent=aa)
    aba = SoftmaxNode("aba", parent=ab)
    abb = SoftmaxNode("abb", parent=ab)
    
    baa = SoftmaxNode("baa", parent=ba)
    bab = SoftmaxNode("bab", parent=ba)
    bba = SoftmaxNode("bba", parent=bb)
    bbb = SoftmaxNode("bbb", parent=bb)

    targets = [aaa,aab,aba, abb, baa, bab, bba, bbb]

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent
    node_predictions = greedy_predictions(prediction_tensor=predictions, root=root, max_depth=1)
    assert [str(node) for node in node_predictions] == ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']

    node_predictions = greedy_predictions(prediction_tensor=predictions, root=root, max_depth=2)
    assert [str(node) for node in node_predictions] == ['aa', 'aa', 'ab', 'ab', 'ba', 'ba', 'bb', 'bb']
