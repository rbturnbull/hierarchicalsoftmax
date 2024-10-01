import torch
import pytest
from hierarchicalsoftmax.nodes import SoftmaxNode, IndexNotSetError
from hierarchicalsoftmax.inference import (
    ShapeError, 
    greedy_predictions,
    node_probabilities, 
    leaf_probabilities, 
    render_probabilities,
)
from pathlib import Path
import tempfile

from .util import depth_two_tree_and_targets, depth_three_tree_and_targets, depth_three_tree_and_targets_only_child

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


def test_node_probabilities():
    root, targets = depth_three_tree_and_targets()

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    probabilities = node_probabilities(prediction_tensor=predictions, root=root)

    assert probabilities.shape == predictions.shape
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0
    assert torch.allclose(probabilities.sum(dim=1), 3.0*torch.ones(len(targets)))


def test_greedy_predictions_threshold():
    root, targets = depth_three_tree_and_targets()

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 1.0
            target = target.parent

    probabilities = node_probabilities(prediction_tensor=predictions, root=root)
    node_predictions = greedy_predictions(prediction_tensor=probabilities, root=root, threshold=0.5)
    assert [str(node) for node in node_predictions] == ["aa", "aa", "ab", "ab", "ba", "ba", "bb", "bb"]

    node_predictions = greedy_predictions(prediction_tensor=probabilities, root=root, threshold=0.1)
    assert [str(node) for node in node_predictions] == ["aaa", "aab", "aba", "abb", "baa", "bab", "bba", "bbb"]

    node_predictions = greedy_predictions(prediction_tensor=probabilities, root=root, threshold=0.70)
    assert [str(node) for node in node_predictions] == ["a", "a", "a", "a", "b", "b", "b", "b"]

    node_predictions = greedy_predictions(prediction_tensor=probabilities, root=root, threshold=0.9)
    assert [str(node) for node in node_predictions] == ["root", "root", "root", "root", "root", "root", "root", "root"]






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


def test_render_probabilities():
    root, targets = depth_three_tree_and_targets()

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        
        # Test DOT generation
        filepaths = [tmpdir/f'test_render_probabilities_{i}.dot' for i in range(len(targets))]
        render_probabilities(prediction_tensor=predictions, root=root, filepaths=filepaths)
        for filepath in filepaths:
            assert filepath.exists()
            assert "digraph tree {" in filepath.read_text()

        # Test PNG rendering
        filepaths = [tmpdir/f'test_render_probabilities_{i}.png' for i in range(len(targets))]
        render_probabilities(prediction_tensor=predictions, root=root, filepaths=filepaths)
        for filepath in filepaths:
            assert filepath.exists()
            with pytest.raises(UnicodeDecodeError):
                filepath.read_text()


def test_render_probabilities_depth_three_tree_and_targets_only_child():
    root, targets = depth_three_tree_and_targets_only_child()

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        
        # Test DOT generation
        filepaths = [tmpdir/f'test_render_probabilities_{i}.dot' for i in range(len(targets))]
        render_probabilities(prediction_tensor=predictions, root=root, filepaths=filepaths)
        for filepath in filepaths:
            assert filepath.exists()
            assert "digraph tree {" in filepath.read_text()

        # Test PNG rendering
        filepaths = [tmpdir/f'test_render_probabilities_{i}.png' for i in range(len(targets))]
        render_probabilities(prediction_tensor=predictions, root=root, filepaths=filepaths)
        for filepath in filepaths:
            assert filepath.exists()
            with pytest.raises(UnicodeDecodeError):
                filepath.read_text()


def test_node_probabilities_shape_error():
    root, targets = depth_two_tree_and_targets()

    root.set_indexes()

    predictions = torch.zeros( (len(targets), root.layer_size + 1) )
    with pytest.raises(ShapeError):
        node_probabilities(prediction_tensor=predictions, root=root)


def test_node_probabilities_index_error():
    root, targets = depth_two_tree_and_targets()

    predictions = torch.zeros( (len(targets), 10) )

    with pytest.raises(IndexNotSetError):
        node_probabilities(prediction_tensor=predictions, root=root)

