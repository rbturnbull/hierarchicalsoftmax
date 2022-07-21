import re
from hierarchicalsoftmax import SoftmaxNode

import itertools

def seven_node_tree():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    return root


def assert_multiline_strings( string1, string2 ):
    lines1 = str(string1).strip().split("\n")
    lines2 = str(string2).strip().split("\n")

    assert len(lines1) == len(lines2)

    for line1, line2 in zip(lines1, lines2):
        assert line1.strip() == line2.strip()


