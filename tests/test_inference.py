import torch
import pytest
from hierarchicalsoftmax.nodes import SoftmaxNode, IndexNotSetError
from hierarchicalsoftmax.inference import greedy_predictions, ShapeError, tree_probabilities, leaf_probabilities

from .util import depth_two_tree_and_targets, depth_three_tree_and_targets

def test_greedy_predictions():
    root, targets = depth_two_tree_and_targets()
    root.set_indexes()

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    node_predictions = greedy_predictions(prediction_tensor=predictions, root=root)

    assert node_predictions == targets


def test_unset_indexes():
    root, targets = depth_two_tree_and_targets()

    predictions = torch.zeros( (len(targets), 10) )

    with pytest.raises(IndexNotSetError):
        greedy_predictions(prediction_tensor=predictions, root=root)


def test_greedy_predictions():
    root, targets = depth_two_tree_and_targets()

    root.set_indexes()

    predictions = torch.zeros( (len(targets), root.layer_size + 1) )
    with pytest.raises(ShapeError):
        node_predictions = greedy_predictions(prediction_tensor=predictions, root=root)


def test_max_depth_simple():
    root, targets = depth_two_tree_and_targets()

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent
    node_predictions = greedy_predictions(prediction_tensor=predictions, root=root, max_depth=1)
    assert [str(node) for node in node_predictions] == ['a', 'a', 'b', 'b']


def test_max_depth_complex():
    root, targets = depth_three_tree_and_targets()

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent
    node_predictions = greedy_predictions(prediction_tensor=predictions, root=root, max_depth=1)
    assert [str(node) for node in node_predictions] == ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']

    node_predictions = greedy_predictions(prediction_tensor=predictions, root=root, max_depth=2)
    assert [str(node) for node in node_predictions] == ['aa', 'aa', 'ab', 'ab', 'ba', 'ba', 'bb', 'bb']


def test_tree_probabilities():
    root, targets = depth_three_tree_and_targets()

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    probabilities = tree_probabilities(prediction_tensor=predictions, root=root)

    assert probabilities.shape == predictions.shape
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0


def test_leaf_probabilities():
    root, targets = depth_three_tree_and_targets()

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    probabilities = leaf_probabilities(prediction_tensor=predictions, root=root)

    assert probabilities.shape == (len(targets), 8)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(len(targets)))

