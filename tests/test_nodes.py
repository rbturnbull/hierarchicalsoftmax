from pathlib import Path
import pytest
from hierarchicalsoftmax.nodes import SoftmaxNode, ReadOnlyError, AlreadyIndexedError
import tempfile
import torch
from .util import depth_two_tree, depth_two_tree_and_targets_three_children, correct_predictions, depth_three_tree_and_targets_only_child
from anytree import PreOrderIter, PostOrderIter, LevelOrderIter, LevelOrderGroupIter, ZigZagGroupIter


def test_simple_tree():
    root = depth_two_tree()
    root.set_indexes()
    assert root.render_equal("""
            root
            ├── a
            │   ├── aa
            │   └── ab
            └── b
                ├── ba
                └── bb    
        """,
        print=True,
    )
    assert root.render_equal("""
            0
            ├── 2
            │   ├── None
            │   └── None
            └── 4
                ├── None
                └── None
        """,
        attr="softmax_start_index"
    )
    assert root.render_equal("""
        2
        ├── 4
        │   ├── None
        │   └── None
        └── 6
            ├── None
            └── None
        """,
        attr="softmax_end_index"
    )

    assert root.layer_size == 6



def test_read_only():
    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)

    assert root.render_equal("""
        root
        └── a
            ├── aa
            └── ab
        """,
    )

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


def test_render_equal_false():
    root = depth_two_tree()
    assert not root.render_equal("""
        root
        ├── a
        │   ├── aa!
        │   └── ab
        └── b
            ├── ba
            └── bb    
    """
    )
    assert not root.render_equal("""
        root
        ├── a
        │   ├── aa
        │   └── ab
        └── b
            ├── ba
            ├── bab
            └── bb    
    """
    )


def test_render_depth_three_tree_and_targets_only_child():
    root, _ = depth_three_tree_and_targets_only_child()
    print(root.render())
    assert root.render_equal("""
        root
        ├── a
        │   └── aa
        │       ├── aaa
        │       └── aab
        └── b
            └── ba
                ├── baa
                └── bab
    """)


def test_render_depth_three_tree_and_targets_layer_size():
    root, _ = depth_three_tree_and_targets_only_child()
    root.set_indexes()
    assert root.layer_size == 6
    assert len(root.node_list) == 8
    assert len(root.node_list_softmax) == 6
    assert [node.name for node in root.node_list_softmax] == ["a", "b", "aaa", "aab", "baa", "bab"]


def test_node_indexes():
    root, targets = depth_two_tree_and_targets_three_children()
    root.set_indexes()

    predictions = correct_predictions(root, targets)

    for node in root.pre_order_iter():
        if node != root:
            assert node.index_in_softmax_layer == node.index_in_parent + node.parent.softmax_start_index
        else:
            assert node.index_in_softmax_layer is None
            assert node.index_in_parent is None
            assert node.parent is None

        if node.is_leaf:
            assert node.softmax_start_index is None
            assert node.softmax_end_index is None
        else:
            cropped = predictions[:, node.softmax_start_index:node.softmax_end_index]    
            assert cropped.shape[-1] == len(node.children)


def test_pre_order_iter():
    root, _ = depth_two_tree_and_targets_three_children()
    assert isinstance(root.pre_order_iter(), PreOrderIter)
    assert [node.name for node in root.pre_order_iter()] == ['root', 'a', 'aa', 'ab', 'ac', 'b', 'ba', 'bb', 'bc']
    assert [node.name for node in root.pre_order_iter(depth=1)] == ['root', 'a', 'b']


def test_post_order_iter():
    root, _ = depth_two_tree_and_targets_three_children()
    assert isinstance(root.post_order_iter(), PostOrderIter)
    assert [node.name for node in root.post_order_iter()] == ['aa', 'ab', 'ac', 'a', 'ba', 'bb', 'bc', 'b', 'root']
    assert [node.name for node in root.post_order_iter(depth=1)] == ['a', 'b', 'root']


def test_level_order_iter():
    root, _ = depth_two_tree_and_targets_three_children()
    assert isinstance(root.level_order_iter(), LevelOrderIter)
    assert [node.name for node in root.level_order_iter()] == ['root', 'a', 'b', 'aa', 'ab', 'ac', 'ba', 'bb', 'bc']
    assert [node.name for node in root.level_order_iter(depth=1)] == ['root', 'a', 'b']


def test_level_order_group_iter():
    root, _ = depth_two_tree_and_targets_three_children()
    assert isinstance(root.level_order_group_iter(), LevelOrderGroupIter)
    assert str(list(root.level_order_group_iter())) == "[(root,), (a, b), (aa, ab, ac, ba, bb, bc)]"
    assert str(list(root.level_order_group_iter(depth=1))) == '[(root,), (a, b)]'


def test_zig_zag_group_iter():
    root, _ = depth_two_tree_and_targets_three_children()
    assert isinstance(root.zig_zag_group_iter(), ZigZagGroupIter)
    assert str(list(root.zig_zag_group_iter())) == '[(root,), (b, a), (aa, ab, ac, ba, bb, bc)]'
    assert str(list(root.zig_zag_group_iter(depth=1))) == '[(root,), (b, a)]'


def test_pre_order_iter_non_root():
    root, _ = depth_two_tree_and_targets_three_children()
    node = root.get_child_by_name("a")
    assert isinstance(node.pre_order_iter(), PreOrderIter)
    assert [n.name for n in node.pre_order_iter()] == ['a', 'aa', 'ab', 'ac']
    assert [n.name for n in node.pre_order_iter(depth=1)] == ['a', 'aa', 'ab', 'ac']


def test_svg():
    root, _ = depth_two_tree_and_targets_three_children()
    output = root.svg()
    assert output.startswith('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    assert output.strip().endswith('</svg>')
    assert '<text text-anchor="middle" x="207"' in output


def test_best_index_in_softmax_layer():
    root, _ = depth_two_tree_and_targets_three_children()
    root.set_indexes()

    assert root.best_index_in_softmax_layer() == None
    for node in root.pre_order_iter():
        if node == root:
            continue
        assert node.best_index_in_softmax_layer() == node.index_in_softmax_layer
