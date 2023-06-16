from pathlib import Path
import pytest
from hierarchicalsoftmax.nodes import SoftmaxNode, ReadOnlyError, AlreadyIndexedError
import tempfile
from .util import depth_two_tree, assert_multiline_strings

def test_simple_tree():
    root = depth_two_tree()
    root.set_indexes()
    assert_multiline_strings( root.render(print=True), """
        root
        ├── a
        │   ├── aa
        │   └── ab
        └── b
            ├── ba
            └── bb    
    """)
    assert_multiline_strings( root.render(attr="softmax_start_index", print=True), """
        0
        ├── 2
        │   ├── None
        │   └── None
        └── 4
            ├── None
            └── None
    """)
    assert_multiline_strings( root.render(attr="softmax_end_index", print=True), """
        2
        ├── 4
        │   ├── None
        │   └── None
        └── 6
            ├── None
            └── None
    """)
    assert root.layer_size == 6



def test_read_only():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)

    assert_multiline_strings( root.render(print=True), """
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
    assert node_names == ['a', 'b', 'aa', 'ab', 'ba', 'bb']
    assert root.node_to_id[a] == 0
    assert root.node_to_id[bb] == 5

    assert root.get_node_ids( [bb, aa, ab] ) == [5,2,3]


def test_already_indexed():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    root.set_indexes()
    with pytest.raises(AlreadyIndexedError):
        root.set_indexes()


def test_get_node_ids_does_index():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    node_ids = root.get_node_ids([aa])
    with pytest.raises(AlreadyIndexedError):
        root.set_indexes()


def test_render_svg():
    root = depth_two_tree()

    with tempfile.NamedTemporaryFile(suffix=".svg") as f:
        path = Path(f.name)
        root.render(filepath=path)
        assert path.exists()
        text = path.read_text()
        assert 3800 < path.stat().st_size < 3900
        assert text.startswith('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')


def test_render_dot():
    root = depth_two_tree()

    with tempfile.NamedTemporaryFile(suffix=".dot") as f:
        path = Path(f.name)
        root.render(filepath=path)
        assert path.exists()
        text = path.read_text()
        assert 180 < path.stat().st_size < 200
        assert text.startswith('digraph tree {\n    "root";\n    "a";\n')
