import torch
from hierarchicalsoftmax.nodes import SoftmaxNode
from hierarchicalsoftmax.metrics import greedy_accuracy, greedy_f1_score, greedy_accuracy_depth_one, greedy_accuracy_depth_two

from .util import depth_two_tree_and_targets, depth_three_tree_and_targets

def test_greedy_accuracy():
    root, targets = depth_two_tree_and_targets()

    root.set_indexes()

    target_tensor = root.get_node_ids_tensor(targets)

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    assert greedy_accuracy(predictions, target_tensor, root=root) > 0.99 

    for target_index, target in enumerate(targets[:2]):
        predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = -20.0

    assert 0.49 < greedy_accuracy(predictions, target_tensor, root=root) < 0.51


def test_greedy_f1_score():
    root, targets = depth_two_tree_and_targets()

    root.set_indexes()

    target_tensor = root.get_node_ids_tensor(targets)

    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    assert greedy_f1_score(predictions, target_tensor, root=root) > 0.99 

    for target_index, target in enumerate(targets[:2]):
        predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = -20.0

    assert 0.49 < greedy_f1_score(predictions, target_tensor, root=root) < 0.51


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

