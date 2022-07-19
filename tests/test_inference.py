import torch
from hierarchicalsoftmax.nodes import SoftmaxNode, ReadOnlyError
from hierarchicalsoftmax.inference import greedy_predictions

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

    predictions = torch.zeros( (len(targets), root.children_softmax_end_index) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    node_predictions = greedy_predictions(root=root, all_predictions=predictions)

    assert node_predictions == targets