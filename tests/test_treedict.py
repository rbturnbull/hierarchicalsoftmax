import pytest
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from collections import Counter
from anytree import RenderTree
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
