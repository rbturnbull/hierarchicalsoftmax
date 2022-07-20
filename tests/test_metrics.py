import torch
from hierarchicalsoftmax.nodes import SoftmaxNode, ReadOnlyError
from hierarchicalsoftmax.metrics import greedy_accuracy

def test_greedy_accuracy():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    root.set_indexes()

    targets = [aa,ba,bb, ab]
    target_tensor = root.get_node_ids_tensor(targets)

    predictions = torch.zeros( (len(targets), root.children_softmax_end_index) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    assert greedy_accuracy(predictions, target_tensor, root=root) > 0.99 

    for target_index, target in enumerate(targets[:2]):
        predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = -20.0

    assert 0.49 < greedy_accuracy(predictions, target_tensor, root=root) < 0.51

