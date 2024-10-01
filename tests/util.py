from hierarchicalsoftmax import SoftmaxNode
import torch

def depth_two_tree_and_targets():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    targets = [aa, ab, ba, bb]

    return root, targets


def depth_two_tree_and_targets_three_children():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    ac = SoftmaxNode("ac", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)
    bc = SoftmaxNode("bc", parent=b)

    targets = [aa, ab, ac, ba, bb, bc]

    return root, targets

def depth_two_tree():
    root, _ = depth_two_tree_and_targets()
    return root


def assert_multiline_strings( string1, string2 ):
    lines1 = str(string1).strip().split("\n")
    lines2 = str(string2).strip().split("\n")

    assert len(lines1) == len(lines2)

    for line1, line2 in zip(lines1, lines2):
        assert line1.strip() == line2.strip()


def depth_three_tree_and_targets():
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

    return root, targets


def depth_three_tree_and_targets_only_child():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)

    aaa = SoftmaxNode("aaa", parent=aa)
    aab = SoftmaxNode("aab", parent=aa)
    
    baa = SoftmaxNode("baa", parent=ba)
    bab = SoftmaxNode("bab", parent=ba)

    targets = [aaa,aab,baa, bab,]

    return root, targets


def depth_three_tree():
    root, _ = depth_three_tree_and_targets()
    return root


def correct_predictions(root, targets):
    predictions = torch.zeros( (len(targets), root.layer_size) )
    for target_index, target in enumerate(targets):
        while target.parent:
            predictions[ target_index, target.parent.softmax_start_index + target.index_in_parent ] = 20.0
            target = target.parent

    return predictions