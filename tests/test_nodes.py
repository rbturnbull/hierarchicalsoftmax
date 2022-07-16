import re
import pytest
from hierarchicalsoftmax.nodes import SoftmaxNode, ReadOnlyError

def assert_rendered( rendered, target ):
    # Remove indents
    indent = re.match(r"(\n\s*)", target)
    if indent:
        target = target.replace(indent.group(1), "\n")

    assert str(rendered) == target.strip()


def test_simple_tree():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    root.set_indexes()
    assert_rendered( root.render(print=True), """
        root
        ├── a
        │   ├── aa
        │   └── ab
        └── b
            ├── ba
            └── bb    
    """)
    assert_rendered( root.render(attr="softmax_start_index", print=True), """
        0
        ├── 2
        │   ├── None
        │   └── None
        └── 4
            ├── None
            └── None
    """)
    assert_rendered( root.render(attr="softmax_end_index", print=True), """
        2
        ├── 4
        │   ├── None
        │   └── None
        └── 6
            ├── None
            └── None
    """)
    assert root.children_softmax_end_index == 6



def test_read_only():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)

    assert_rendered( root.render(print=True), """
        root
        └── a
            ├── aa
            └── ab
    """)
    
    root.set_indexes()

    with pytest.raises(ReadOnlyError):
        b = SoftmaxNode("b", parent=root)

    with pytest.raises(ReadOnlyError):
        a.parent = None


def test_get_child_by_name():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)

    assert root.get_child_by_name("a") is a
    assert a.get_child_by_name("a") is None
    assert aa.get_child_by_name("root") is None
    assert a.get_child_by_name("aa") is aa

    # Test detachment
    aa.parent = None
    assert a.get_child_by_name("aa") is None


def test_node_list():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    root.set_indexes()

    node_names = [node.name for node in root.node_list]
    assert node_names == ['root', 'a', 'aa', 'ab', 'b', 'ba', 'bb']
    assert root.node_to_id[root] == 0
    assert root.node_to_id[bb] == 6