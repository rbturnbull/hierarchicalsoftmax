import pytest 
import torch
from hierarchicalsoftmax.nodes import IndexNotSetError
from hierarchicalsoftmax.metrics import (
    greedy_accuracy, 
    greedy_f1_score, 
    greedy_accuracy_depth_one, 
    greedy_accuracy_depth_two,
    greedy_accuracy_parent,
    greedy_precision,
    greedy_recall,
    depth_accurate,
    GreedyAccuracyTorchMetric,
    GreedyAccuracy,
    RankAccuracyTorchMetric,
)
from hierarchicalsoftmax.inference import ShapeError
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

    metric = GreedyAccuracy(root=root)
    assert_allclose(metric(predictions, target_tensor), 0.75)


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

    depth_one = GreedyAccuracy(root=root, max_depth=1, name="depth_one")
    assert 0.99 < depth_one(predictions_rearranged, target_tensor)     
    depth_two = GreedyAccuracy(root=root, max_depth=2, name="depth_two")
    assert depth_two(predictions_rearranged, target_tensor) < 0.01

    assert depth_one.name == "depth_one"
    assert depth_one.__name__ == "depth_one"
    assert depth_two.name == "depth_two"
    assert depth_two.__name__ == "depth_two"


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


@pytest.fixture
def setup_depth_three_tests():
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

    return predictions, target_tensor, root


def test_greedy_accuracy_parent(setup_depth_three_tests):
    predictions, target_tensor, root = setup_depth_three_tests

    assert 0.874 < greedy_accuracy_parent(predictions, target_tensor, root=root) < 0.876


def test_depth_accurate(setup_depth_three_tests):
    predictions, target_tensor, root = setup_depth_three_tests

    result = depth_accurate(predictions, target_tensor, root=root)
    assert (result == torch.tensor([2, 1, 3, 3, 3, 3, 3, 2])).all()


def test_depth_accurate_max_depth(setup_depth_three_tests):
    predictions, target_tensor, root = setup_depth_three_tests

    result = depth_accurate(predictions, target_tensor, root=root, max_depth=2)
    assert (result == torch.tensor([2, 1, 2, 2, 2, 2, 2, 2])).all()


def test_depth_accuracte_set_indexes():
    root, targets = depth_two_tree_and_targets()
    predictions = torch.zeros( (len(targets), 6) )
    targets = torch.zeros( (len(targets),) )
    with pytest.raises(IndexNotSetError):
        node_predictions = depth_accurate(predictions, targets, root=root)



def test_depth_accuracte_shape_error():
    root, targets = depth_two_tree_and_targets()
    root.set_indexes()
    predictions = torch.zeros( (len(targets), 7) )
    targets = torch.zeros( (len(targets),) )
    with pytest.raises(ShapeError):
        node_predictions = depth_accurate(predictions, targets, root=root)



def test_greedy_accuracy_initialization(setup_depth_three_tests):
    predictions, target_tensor, root = setup_depth_three_tests
    metric = GreedyAccuracyTorchMetric(root=root, max_depth=2)
    assert metric.root == root
    assert metric.max_depth == 2
    assert metric.name == 'greedy_accuracy_2'
    assert hasattr(metric, 'total')
    assert hasattr(metric, 'correct')


def test_greedy_accuracy_update(setup_depth_three_tests):
    predictions, target_tensor, root = setup_depth_three_tests
    metric = GreedyAccuracyTorchMetric(root=root, max_depth=2)

    # Patch the depth_accurate function    
    metric.update(predictions, target_tensor)
    
    assert metric.total.item() == 8
    assert metric.correct.item() == 7


def test_greedy_accuracy_compute(setup_depth_three_tests):
    predictions, target_tensor, root = setup_depth_three_tests
    ranks = {1: 'rank_1', 2: 'rank_2', 3: 'rank_3'}
    metric = GreedyAccuracyTorchMetric(root=root, max_depth=2)
    
    # Patch the depth_accurate function    
    metric.update(predictions, target_tensor)
    
    result = metric.compute()
    assert pytest.approx(result.item()) == 0.875
    

def test_rank_accuracy_initialization(setup_depth_three_tests):
    predictions, target_tensor, root = setup_depth_three_tests
    ranks = {1: 'rank_1', 2: 'rank_2', 3: 'rank_3'}
    metric = RankAccuracyTorchMetric(root=root, ranks=ranks)
    assert metric.root == root
    assert metric.ranks == ranks
    assert metric.name == 'rank_accuracy'
    assert hasattr(metric, 'total')
    for rank_name in ranks.values():
        assert hasattr(metric, rank_name)
        assert getattr(metric, rank_name).item() == 0


def test_rank_accuracy_update(setup_depth_three_tests):
    predictions, target_tensor, root = setup_depth_three_tests
    ranks = {1: 'rank_1', 2: 'rank_2', 3: 'rank_3'}
    metric = RankAccuracyTorchMetric(root=root, ranks=ranks)
    
    # Patch the depth_accurate function    
    metric.update(predictions, target_tensor)
    
    assert metric.total.item() == 8
    assert metric.rank_2.item() == 7
    assert metric.rank_3.item() == 5


def test_rank_accuracy_compute(setup_depth_three_tests):
    predictions, target_tensor, root = setup_depth_three_tests
    ranks = {1: 'rank_1', 2: 'rank_2', 3: 'rank_3'}
    metric = RankAccuracyTorchMetric(root=root, ranks=ranks)
    
    # Patch the depth_accurate function    
    metric.update(predictions, target_tensor)
    
    result = metric.compute()
    
    assert pytest.approx(result['rank_1'].item()) == 1.0
    assert pytest.approx(result['rank_2'].item()) == 0.875
    assert pytest.approx(result['rank_3'].item()) == 0.625


def test_rank_accuracy_compute_tuple(setup_depth_three_tests):
    predictions, target_tensor, root = setup_depth_three_tests
    ranks = {1: 'rank_1', 2: 'rank_2', 3: 'rank_3'}
    metric = RankAccuracyTorchMetric(root=root, ranks=ranks)
    
    predictions_tuple = (predictions, )

    # Patch the depth_accurate function    
    metric.update(predictions_tuple, target_tensor)
    
    result = metric.compute()
    
    assert pytest.approx(result['rank_1'].item()) == 1.0
    assert pytest.approx(result['rank_2'].item()) == 0.875
    assert pytest.approx(result['rank_3'].item()) == 0.625


def test_rank_accuracy_apply(setup_depth_three_tests):
    predictions, target_tensor, root = setup_depth_three_tests
    ranks = {1: 'rank_1', 2: 'rank_2', 3: 'rank_3'}
    metric = RankAccuracyTorchMetric(root=root, ranks=ranks)
    
    # Patch the depth_accurate function    
    metric.update(predictions, target_tensor)
    
    result = metric.compute()
    
    assert pytest.approx(result['rank_1'].item()) == 1.0
    assert pytest.approx(result['rank_2'].item()) == 0.875
    assert pytest.approx(result['rank_3'].item()) == 0.625

    def multiply_by_10(tensor):
        return tensor * 10

    metric._apply(multiply_by_10)

    result = metric.compute()

    assert pytest.approx(result['rank_1'].item()) == 10.0