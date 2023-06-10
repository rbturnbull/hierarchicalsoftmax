import torch
from hierarchicalsoftmax.nodes import SoftmaxNode
from hierarchicalsoftmax.metrics import (
    greedy_accuracy, 
    greedy_f1_score, 
    greedy_accuracy_depth_one, 
    greedy_accuracy_depth_two,
    greedy_accuracy_parent,
    greedy_precision,
    greedy_recall,
)
from torch.testing import assert_allclose

from .util import depth_two_tree_and_targets, depth_three_tree_and_targets, depth_two_tree_and_targets_three_children

def test_greedy_accuracy():
    root, targets = depth_two_tree_and_targets_three_children()

    root.set_indexes()
    targets *= 2
    target_tensor = root.get_node_ids_tensor(targets)

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    assert_allclose(greedy_accuracy(predictions, target_tensor, root=root), 1.0)

    for target_index, target in enumerate(targets[:3]):
        predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = -20.0

    assert_allclose(greedy_accuracy(predictions, target_tensor, root=root), 0.75)


def test_greedy_f1_score():
    root, targets = depth_two_tree_and_targets_three_children()

    root.set_indexes()
    targets *= 2
    target_tensor = root.get_node_ids_tensor(targets)

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    assert_allclose(greedy_f1_score(predictions, target_tensor, root=root), 1.0)

    for target_index, target in enumerate(targets[:3]):
        predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = -20.0

    assert_allclose(greedy_f1_score(predictions, target_tensor, root=root), 0.7611111111111111)


def test_greedy_precision():
    root, targets = depth_two_tree_and_targets_three_children()
    root.set_indexes()
    targets *= 2
    target_tensor = root.get_node_ids_tensor(targets)

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    assert_allclose(greedy_precision(predictions, target_tensor, root=root), 1.0)

    for target_index, target in enumerate(targets[:3]):
        predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = -20.0

    assert_allclose(greedy_precision(predictions, target_tensor, root=root), 0.8055555555555555)


def test_greedy_recall():
    root, targets = depth_two_tree_and_targets_three_children()

    root.set_indexes()

    target_tensor = root.get_node_ids_tensor(targets)

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    assert_allclose(greedy_recall(predictions, target_tensor, root=root), 1.0)

    for target_index, target in enumerate(targets[:3]):
        predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = -20.0

    assert_allclose(greedy_recall(predictions, target_tensor, root=root), 0.5)


def test_greedy_accuracy_max_depth_simple():
    root, targets = depth_two_tree_and_targets()

    root.set_indexes()
    target_tensor = root.get_node_ids_tensor(targets)

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    # rearrange predictions so that depth 1 is OK but depth 2 is incorrect
    predictions_rearranged = torch.zeros_like(predictions)
    predictions_rearranged[0] = predictions[1]
    predictions_rearranged[1] = predictions[0]
    predictions_rearranged[2] = predictions[3]
    predictions_rearranged[3] = predictions[2]

    assert 0.99 < greedy_accuracy_depth_one(predictions_rearranged, target_tensor, root=root) 
    assert greedy_accuracy_depth_two(predictions_rearranged, target_tensor, root=root) < 0.01
    assert greedy_accuracy(predictions_rearranged, target_tensor, root=root) < 0.01


def test_greedy_accuracy_max_depth_complex():
    root, targets = depth_three_tree_and_targets()

    root.set_indexes()
    target_tensor = root.get_node_ids_tensor(targets)

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    # rearrange predictions so that depth 1 is OK but depth 2 is incorrect
    predictions_rearranged = torch.zeros_like(predictions)
    predictions_rearranged[0] = predictions[3]
    predictions_rearranged[1] = predictions[2]
    predictions_rearranged[2] = predictions[1]
    predictions_rearranged[3] = predictions[0]
    predictions_rearranged[4] = predictions[7]
    predictions_rearranged[5] = predictions[6]
    predictions_rearranged[6] = predictions[5]
    predictions_rearranged[7] = predictions[4]

    assert 0.99 < greedy_accuracy_depth_one(predictions_rearranged, target_tensor, root=root) 
    assert greedy_accuracy_depth_two(predictions_rearranged, target_tensor, root=root) < 0.01
    assert greedy_accuracy(predictions_rearranged, target_tensor, root=root) < 0.01


def test_greedy_accuracy_parent():
    root, targets = depth_three_tree_and_targets()

    root.set_indexes()
    target_tensor = root.get_node_ids_tensor(targets)

    # set up predictions
    prediction_nodes = targets.copy()
    aaa, aab, aba, abb, baa, bab, bba, bbb = targets
    prediction_nodes[0] = aab # correct parent
    prediction_nodes[7] = bba # correct parent
    prediction_nodes[1] = aba # incorrect parent

    predictions = torch.zeros( (len(prediction_nodes), root.layer_size) )
    for prediction_index, prediction in enumerate(prediction_nodes):
        while prediction.parent:
            predictions[ prediction_index, prediction.parent.softmax_start_index + prediction.index_in_parent ] = 20.0
            prediction = prediction.parent

    assert 0.874 < greedy_accuracy_parent(predictions, target_tensor, root=root) < 0.876
