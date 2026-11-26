import torch
import pytest
from hierarchicalsoftmax.nodes import SoftmaxNode, IndexNotSetError
from hierarchicalsoftmax.inference import (
    ShapeError, 
    greedy_predictions,
    node_probabilities, 
    leaf_probabilities, 
    render_probabilities,
    greedy_lineage_probabilities,
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


def test_greedy_lineage_probabilities():
    root, targets = depth_two_tree_and_targets()
    root.set_indexes()

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 1.0
            target = target.parent

    results = greedy_lineage_probabilities(prediction_tensor=predictions, root=root)
    for target, result, in zip(targets, results):
        prediction_node, probability = result[-1]
        assert isinstance(prediction_node, SoftmaxNode)
        assert prediction_node == target
        assert isinstance(probability, float)
        for node, probability in result:
            assert isinstance(node, SoftmaxNode)
            assert isinstance(probability, float)
            assert 0.5 <= probability <= 1.0


def test_greedy_lineage_probabilities_only_child():
    root, targets = depth_three_tree_and_targets_only_child()
    root.set_indexes()

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            if len(target.parent.children) > 1:
                predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    results = greedy_lineage_probabilities(prediction_tensor=predictions, root=root)
    for target, result, in zip(targets, results):
        prediction_node, probability = result[-1]
        assert isinstance(prediction_node, SoftmaxNode)
        assert prediction_node == target
        assert isinstance(probability, float)
        for node, probability in result:
            assert isinstance(node, SoftmaxNode)
            assert isinstance(probability, float)
            assert 0.5 <= probability <= 1.0



def test_greedy_lineage_probabilities_threshold():
    root, targets = depth_two_tree_and_targets()
    root.set_indexes()

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 1.0
            target = target.parent

    results = greedy_lineage_probabilities(prediction_tensor=(predictions,), root=root, threshold=0.7)
    for target, result, in zip(targets, results):
        prediction_node, probability = result[-1]
        assert isinstance(prediction_node, SoftmaxNode)
        assert prediction_node == target.parent
        assert isinstance(probability, float)
        for node, probability in result:
            assert isinstance(node, SoftmaxNode)
            assert isinstance(probability, float)
            assert 0.7 <= probability <= 1.0


def test_greedy_lineage_probabilities_max_depth():
    root, targets = depth_two_tree_and_targets()
    root.set_indexes()

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 1.0
            target = target.parent

    results = greedy_lineage_probabilities(prediction_tensor=(predictions,), root=root, max_depth=1)
    for target, result, in zip(targets, results):
        prediction_node, probability = result[-1]
        assert isinstance(prediction_node, SoftmaxNode)
        assert prediction_node == target.parent
        assert isinstance(probability, float)
        assert len(result) == 1
        for node, probability in result:
            assert isinstance(node, SoftmaxNode)
            assert isinstance(probability, float)
            assert 0.7 <= probability <= 1.0


def test_greedy_predictions_tuple():
    root, targets = depth_two_tree_and_targets()
    root.set_indexes()

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    predictions_tuple = (predictions, )
    node_predictions = greedy_predictions(prediction_tensor=predictions_tuple, root=root)

    assert node_predictions == targets


def test_unset_indexes():
    root, targets = depth_two_tree_and_targets()

    predictions = torch.zeros( (len(targets), 10) )

    with pytest.raises(IndexNotSetError):
        greedy_predictions(prediction_tensor=predictions, root=root)


def test_greedy_predictions_shape_error():
    root, targets = depth_two_tree_and_targets()

    root.set_indexes()

    predictions = torch.zeros( (len(targets), root.layer_size + 1) )
    with pytest.raises(ShapeError):
        greedy_predictions(prediction_tensor=predictions, root=root)


def test_greedy_lineage_probabilities_shape_error():
    root, targets = depth_two_tree_and_targets()

    root.set_indexes()

    predictions = torch.zeros( (len(targets), root.layer_size + 1) )
    with pytest.raises(ShapeError):
        greedy_lineage_probabilities(prediction_tensor=predictions, root=root)


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
            if len(target.parent.children) > 1:
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


def test_greedy_lineage_probabilities_index_error():
    root, targets = depth_two_tree_and_targets()

    predictions = torch.zeros( (len(targets), 10) )

    with pytest.raises(IndexNotSetError):
        greedy_lineage_probabilities(prediction_tensor=predictions, root=root)



def test_greedy_predictions_threshold_single_child_last():
    """
    A single-child node that comes last in pre-order has softmax_start_index == layer_size.
    The threshold check must not index into the predictions for it (previously raised IndexError).
    """
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    a1 = SoftmaxNode("a1", parent=a)
    a2 = SoftmaxNode("a2", parent=a)
    b = SoftmaxNode("b", parent=root)
    c = SoftmaxNode("c", parent=b)
    root.set_indexes()

    assert b.softmax_start_index == root.layer_size

    probabilities = torch.zeros((1, root.layer_size))
    probabilities[0, a.index_in_softmax_layer] = 0.1
    probabilities[0, b.index_in_softmax_layer] = 0.9
    probabilities[0, a1.index_in_softmax_layer] = 0.05
    probabilities[0, a2.index_in_softmax_layer] = 0.05

    node_predictions = greedy_predictions(prediction_tensor=probabilities, root=root, threshold=0.7)
    assert node_predictions == [c]


def test_greedy_predictions_threshold_single_child_not_last():
    """
    A single-child node that is not last in pre-order shares its softmax_start_index with another node's slot.
    The threshold check must not read that unrelated value and stop the descent early.
    """
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    c = SoftmaxNode("c", parent=a)
    b = SoftmaxNode("b", parent=root)
    b1 = SoftmaxNode("b1", parent=b)
    b2 = SoftmaxNode("b2", parent=b)
    root.set_indexes()

    # The single-child node's start index points at another node's slot
    assert a.softmax_start_index == b1.index_in_softmax_layer

    probabilities = torch.zeros((1, root.layer_size))
    probabilities[0, a.index_in_softmax_layer] = 0.9
    probabilities[0, b.index_in_softmax_layer] = 0.1
    probabilities[0, b1.index_in_softmax_layer] = 0.05
    probabilities[0, b2.index_in_softmax_layer] = 0.05

    node_predictions = greedy_predictions(prediction_tensor=probabilities, root=root, threshold=0.7)
    assert node_predictions == [c]
