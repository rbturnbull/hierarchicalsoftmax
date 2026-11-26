import pytest
import pickle
from pathlib import Path
from unittest.mock import patch
from tempfile import TemporaryDirectory
from collections import Counter
from anytree import RenderTree
from hierarchicalsoftmax import SoftmaxNode
from hierarchicalsoftmax.treedict import TreeDict, NodeDetail, AlreadyExists, app
from typer.testing import CliRunner

from .util import depth_three_tree


def test_node_detail_pickle():
    detail = NodeDetail(partition=1, node_id=5)
    pickled = pickle.dumps(detail)
    loaded = pickle.loads(pickled)
    assert loaded.partition == 1
    assert loaded.node_id == 5
    assert loaded.node is None


def test_add_and_retrieve_node():
    root = depth_three_tree()
    leaf = root.leaves[0]
    treedict = TreeDict(root)
    treedict.add("a", leaf, 0)
    assert treedict["a"].node == leaf
    assert treedict["a"].partition == 0
    assert treedict.node("a") == leaf


def test_already_exists_exception():
    root = depth_three_tree()
    leaf = root.leaves[0]
    treedict = TreeDict(root)
    treedict.add("a", leaf, 0)
    with pytest.raises(AlreadyExists):
        treedict.add("a", root, 0)


def test_set_indexes():
    root = depth_three_tree()
    leaf = root.leaves[0]
    treedict = TreeDict(root)
    treedict.add("a", leaf, 0)
    treedict.set_indexes()
    assert treedict["a"].node_id == root.node_to_id[leaf]


def test_keys_in_partition():
    root = depth_three_tree()
    leaf = root.leaves[0]
    treedict = TreeDict(root)
    treedict.add("a", leaf, 0)
    treedict.add("b", leaf, 1)
    assert list(treedict.keys_in_partition(0)) == ["a"]
    assert list(treedict.keys_in_partition(1)) == ["b"]


def test_truncate():
    root = depth_three_tree()
    leaf = root.leaves[0]
    treedict = TreeDict(root)
    treedict.add("key", leaf, 0)
    truncated = treedict.truncate(2)
    assert truncated["key"].node.name == "a"
    assert truncated["key"].partition == 0
    assert truncated.classification_tree.render_equal("""
            root
            ├── a
            └── b
        """,
        print=True,
    )

    truncated = treedict.truncate(3)
    assert truncated["key"].node.name == "aa"
    assert truncated["key"].partition == 0
    assert truncated.classification_tree.render_equal("""
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



def test_add_counts():
    root = depth_three_tree()
    leaf = root.leaves[0]
    treedict = TreeDict(root)
    treedict.add("a", leaf, 0)
    treedict.add_counts()
    assert leaf.count == 1
    assert leaf.parent.count == 0  # Only leaf got a count


def test_add_partition_counts():
    root = depth_three_tree()
    leaf = root.leaves[0]
    treedict = TreeDict(root)
    treedict.add("a", leaf, 0)
    treedict.add("b", leaf, 1)
    treedict.add_partition_counts()
    assert leaf.partition_counts == Counter({0: 1, 1: 1})


def test_render():
    root = depth_three_tree()
    leaf = root.leaves[0]
    treedict = TreeDict(root)
    treedict.add("a", leaf, 0)

    rendered = treedict.render()
    assert isinstance(rendered, RenderTree)


def test_render_count():
    root = depth_three_tree()
    leaf = root.leaves[0]
    treedict = TreeDict(root)
    treedict.add("a", leaf, 0)

    rendered = treedict.render(count=True)
    assert "aaa (1)" in str(rendered)


def test_render_partition_counts():
    root = depth_three_tree()
    leaf = root.leaves[0]
    leaf2 = root.leaves[1]
    treedict = TreeDict(root)
    treedict.add("a", leaf, 0)
    treedict.add("b", leaf, 3)
    treedict.add("c", leaf2, 2)

    rendered = treedict.render(partition_counts=True)
    assert "aaa 0->1; 3->1" in str(rendered)
    assert "aab 2->1" in str(rendered)


def test_sunburst():
    root = depth_three_tree()
    leaf = root.leaves[0]
    treedict = TreeDict(root)
    treedict.add("a", leaf, 0)

    fig = treedict.sunburst()
    assert fig.data[0].type == "sunburst"
    assert any("labels" in d for d in fig.to_plotly_json()["data"])


def test_save_and_load():
    root = depth_three_tree()
    leaf = root.leaves[0]
    treedict = TreeDict(root)
    treedict.add("a", leaf, 0)

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "tree.pkl"
        treedict.save(path)
        loaded = TreeDict.load(path)
        assert isinstance(loaded, TreeDict)
        assert "a" in loaded


def test_keys_to_file(tmp_path):
    root = depth_three_tree()
    leaf = root.leaves[0]
    treedict = TreeDict(root)
    treedict.add("a", leaf, 0)
    file = tmp_path / "keys.txt"
    treedict.keys_to_file(file)
    contents = file.read_text().strip()
    assert contents == "a"


def test_pickle_tree(tmp_path):
    root = depth_three_tree()
    treedict = TreeDict(root)
    output = tmp_path / "tree_only.pkl"
    treedict.pickle_tree(output)
    tree = pickle.loads(output.read_bytes())
    assert tree.name == "root"



runner = CliRunner()


def make_and_save_treedict(tmp_path):
    root = depth_three_tree()
    leaf = root.leaves[0]
    treedict = TreeDict(root)
    treedict.add("a", leaf, 0)
    treedict.add("b", leaf, 1)
    path = tmp_path / "tree.pkl"
    treedict.save(path)
    return path, treedict


def test_app_keys_all(tmp_path):
    path, _ = make_and_save_treedict(tmp_path)
    result = runner.invoke(app, ["keys", str(path)])
    assert result.exit_code == 0
    assert "a" in result.output
    assert "b" in result.output


def test_app_keys_partition(tmp_path):
    path, _ = make_and_save_treedict(tmp_path)
    result = runner.invoke(app, ["keys", str(path), "--partition", "0"])
    assert result.exit_code == 0
    assert "a" in result.output
    assert "b" not in result.output


def test_app_csv(tmp_path):
    path, _ = make_and_save_treedict(tmp_path)
    csv_path = tmp_path / "treedict.csv"
    result = runner.invoke(app, ["csv", str(path), str(csv_path)])
    assert result.exit_code == 0
    assert csv_path.exists()
    contents = csv_path.read_text()
    assert contents == 'key,node,partition\na,aaa,0\nb,aaa,1\n'


def test_app_render(tmp_path):
    path, _ = make_and_save_treedict(tmp_path)
    out_path = tmp_path / "render.txt"
    result = runner.invoke(app, ["render", str(path), "--output", str(out_path)])
    assert result.exit_code == 0
    assert out_path.exists()


def test_app_render_partition_counts(tmp_path):
    path, treedict = make_and_save_treedict(tmp_path)
    result = runner.invoke(app, ["render", str(path), "--partition-counts"])
    assert result.exit_code == 0


def test_app_count(tmp_path):
    path, _ = make_and_save_treedict(tmp_path)
    result = runner.invoke(app, ["count", str(path)])
    assert result.exit_code == 0
    assert "2" in result.output


def test_app_sunburst_html(tmp_path):
    path, _ = make_and_save_treedict(tmp_path)
    html_path = tmp_path / "sunburst.html"
    result = runner.invoke(app, ["sunburst", str(path), "--output", str(html_path)])
    assert result.exit_code == 0
    assert html_path.exists()
    assert "html" in html_path.read_text().lower()


def test_app_truncate(tmp_path):
    path, _ = make_and_save_treedict(tmp_path)
    out_path = tmp_path / "truncated.pkl"
    result = runner.invoke(app, ["truncate", str(path), "2", str(out_path)])
    assert result.exit_code == 0
    assert out_path.exists()
    loaded = TreeDict.load(out_path)
    assert "a" in loaded


def test_app_layer_size(tmp_path):
    path, treedict = make_and_save_treedict(tmp_path)
    treedict.save(path)

    result = runner.invoke(app, ["layer-size", str(path)])
    assert result.exit_code == 0
    assert "14" == result.output.strip()


def test_app_pickle_tree(tmp_path):
    path, _ = make_and_save_treedict(tmp_path)
    out_path = tmp_path / "tree_only.pkl"
    result = runner.invoke(app, ["pickle-tree", str(path), str(out_path)])
    assert result.exit_code == 0
    tree = pickle.loads(out_path.read_bytes())
    assert tree.name == "root"


def test_app_sunburst_show(tmp_path):
    path, _ = make_and_save_treedict(tmp_path)

    with patch("plotly.graph_objects.Figure.show") as mock_show:
        result = runner.invoke(app, ["sunburst", str(path), "--show"])
        assert result.exit_code == 0
        mock_show.assert_called_once()


def test_set_partition():
    root = depth_three_tree()
    leaf = root.leaves[0]
    treedict = TreeDict(root)
    treedict.add("NC_1", leaf, 0)
    treedict.add("NC_2", leaf, 1)
    treedict.add("NZ_1", leaf, 2)

    treedict.set_partition("NC_*", 5)
    assert set(treedict.keys(partition=5)) == {"NC_1", "NC_2"}
    assert treedict["NZ_1"].partition == 2


def test_render_alphabetical():
    root = SoftmaxNode("root")
    SoftmaxNode("b", parent=root)
    SoftmaxNode("a", parent=root)
    treedict = TreeDict(root)

    unsorted_lines = str(treedict.render()).splitlines()
    assert [line.split()[-1] for line in unsorted_lines[1:]] == ["b", "a"]

    sorted_lines = str(treedict.render(alphabetical=True)).splitlines()
    assert [line.split()[-1] for line in sorted_lines[1:]] == ["a", "b"]


def test_truncate_preserves_subclass():
    class MyTreeDict(TreeDict):
        pass

    root = depth_three_tree()
    treedict = MyTreeDict(root)
    treedict.add("a", root.leaves[0], 0)
    truncated = treedict.truncate(2)
    assert type(truncated) is MyTreeDict
    assert truncated.node("a").name == "a"
    # The original tree is not modified
    assert treedict.node("a") is root.leaves[0]
    assert len(root.leaves[0].ancestors) == 3


def test_merge(tmp_path):
    treedict = TreeDict()
    bacteria = SoftmaxNode("Bacteria", parent=treedict.classification_tree)
    shared = SoftmaxNode("Shared", parent=bacteria)
    treedict.add("existing", shared, 0)
    treedict.add("duplicate", bacteria, 0)

    other = TreeDict()
    other_bacteria = SoftmaxNode("Bacteria", parent=other.classification_tree)
    other_shared = SoftmaxNode("Shared", parent=other_bacteria)
    new_child = SoftmaxNode("New", parent=other_bacteria)
    virus = SoftmaxNode("Virus", parent=other.classification_tree)
    different_parent = SoftmaxNode("Shared", parent=virus)
    SoftmaxNode("Unused", parent=virus)
    other.add("matching", other_shared, 1)
    other.add("new", new_child, 2)
    other.add("different-parent", different_parent, 3)
    other.add("duplicate", virus, 4)

    # Exercise read-only indexed trees and details containing only node IDs.
    treedict.save(tmp_path / "self.pkl")
    other.save(tmp_path / "other.pkl")
    treedict = TreeDict.load(tmp_path / "self.pkl")
    other = TreeDict.load(tmp_path / "other.pkl")
    existing_node = treedict.node("existing")
    treedict.merge(other)

    assert set(treedict) == {"existing", "duplicate", "matching", "new", "different-parent"}
    assert treedict.node("existing") is existing_node
    assert treedict.node("matching") is existing_node
    assert treedict.node("different-parent") is not existing_node
    assert treedict.node("different-parent").parent.name == "Virus"
    assert treedict.node("new").parent is existing_node.parent
    assert treedict.node("duplicate").name == "Virus"
    assert [treedict[key].partition for key in
            ("existing", "matching", "new", "different-parent", "duplicate")] == [0, 1, 2, 3, 4]
    assert treedict.classification_tree.render_equal("""
        root
        ├── Bacteria
        │   ├── Shared
        │   └── New
        └── Virus
            ├── Shared
            └── Unused
    """)
    assert other.node("matching").root is other.classification_tree
    assert other["matching"].node is None
    assert len(other.classification_tree.descendants) == 6

    treedict.save(tmp_path / "merged.pkl")
    restored = TreeDict.load(tmp_path / "merged.pkl")
    for key in treedict:
        assert restored[key].partition == treedict[key].partition
        assert [node.name for node in restored.node(key).path] == [
            node.name for node in treedict.node(key).path]


def test_merge_different_roots():
    treedict = TreeDict()
    other = TreeDict(SoftmaxNode("other-root"))
    with pytest.raises(ValueError, match="different root names"):
        treedict.merge(other)


def test_app_merge(tmp_path):
    treedict = TreeDict()
    bacteria = SoftmaxNode("Bacteria", parent=treedict.classification_tree)
    treedict.add("existing", bacteria, 0)
    treedict.add("duplicate", bacteria, 0)

    other = TreeDict()
    other_bacteria = SoftmaxNode("Bacteria", parent=other.classification_tree)
    virus = SoftmaxNode("Virus", parent=other.classification_tree)
    other.add("matching", other_bacteria, 1)
    other.add("new", virus, 2)
    other.add("duplicate", virus, 3)

    treedict_path = tmp_path / "self.pkl"
    other_path = tmp_path / "other.pkl"
    output = tmp_path / "output" / "merged.pkl"
    treedict.save(treedict_path)
    other.save(other_path)
    input_bytes = [path.read_bytes() for path in (treedict_path, other_path)]

    result = runner.invoke(app, ["merge", str(treedict_path), str(other_path), str(output)])

    assert result.exit_code == 0, (result.output, result.exception)
    merged = TreeDict.load(output)
    assert set(merged) == {"existing", "matching", "new", "duplicate"}
    assert merged.node("matching") is merged.node("existing")
    assert merged.node("new").name == "Virus"
    assert merged.node("duplicate") is merged.node("new")
    assert [merged[key].partition for key in
            ("existing", "matching", "new", "duplicate")] == [0, 1, 2, 3]
    assert merged.classification_tree.render_equal("""
        root
        ├── Bacteria
        └── Virus
    """)
    assert [path.read_bytes() for path in (treedict_path, other_path)] == input_bytes


def test_app_set_partition(tmp_path):
    path, _ = make_and_save_treedict(tmp_path)
    output = tmp_path / "partitioned.pkl"
    result = runner.invoke(app, ["set-partition", str(path), "a*", "9", str(output)])
    assert result.exit_code == 0, (result.output, result.exception)
    loaded = TreeDict.load(output)
    assert list(loaded.keys(partition=9)) == ["a"]
    assert loaded["b"].partition == 1


def test_app_render_alphabetical(tmp_path):
    root = SoftmaxNode("root")
    SoftmaxNode("Virus", parent=root)
    SoftmaxNode("Bacteria", parent=root)
    treedict = TreeDict(root)
    path = tmp_path / "tree.pkl"
    treedict.save(path)
    output = tmp_path / "render.txt"

    result = runner.invoke(app, ["render", str(path), "--no-print-tree", "--output", str(output), "--alphabetical"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert "Virus" not in result.output
    text = output.read_text()
    assert text.index("Bacteria") < text.index("Virus")
