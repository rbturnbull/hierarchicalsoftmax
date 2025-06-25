import copy
from pathlib import Path
from attrs import define, field
from collections import UserDict
import pickle
from collections import Counter
from rich.progress import track
import typer

from .nodes import SoftmaxNode


@define
class NodeDetail:
    """
    Stores metadata for a key in the TreeDict.

    Attributes:
        partition (int): The partition ID this key belongs to.
        node (SoftmaxNode): The node in the classification tree associated with the key.
        node_id (int): The index of the node in the tree (used during pickling).
    """
    partition:int
    node:SoftmaxNode = field(default=None, eq=False)
    node_id:int = None

    def __getstate__(self):
        return (self.partition, self.node_id)

    def __setstate__(self, state):
        self.partition, self.node_id = state
        self.node = None


class AlreadyExists(Exception):
    pass


class TreeDict(UserDict):
    def __init__(self, classification_tree:SoftmaxNode|None=None):
        """
        Initialize a TreeDict.

        Args:
            classification_tree (SoftmaxNode, optional): The root of the classification tree.
                If not provided, a new root node named "root" will be created.
        """
        super().__init__()
        self.classification_tree = classification_tree or SoftmaxNode("root")

    def add(self, key:str, node:SoftmaxNode, partition:int) -> NodeDetail:
        """
        Associate a key with a node and a partition.

        Args:
            key (str): The unique identifier for the item.
            node (SoftmaxNode): The node in the classification tree to associate with the key.
            partition (int): The partition index for the key.

        Raises:
            AlreadyExists: If the key already exists with a different node.

        Returns:
            NodeDetail: The metadata object for the added key.
        """
        assert node.root == self.classification_tree
        if key in self:
            old_node = self.node(key)
            if not node == old_node:
                raise AlreadyExists(f"Accession {key} already exists in TreeDict at node {self.node(key)}. Cannot change to {node}")

        detail = NodeDetail(
            partition=partition,
            node=node,
        )
        self[key] = detail
        return detail

    def set_indexes(self):
        """
        Ensure the tree has assigned node indexes, and record the node_id for each key.
        """
        self.classification_tree.set_indexes_if_unset()
        for detail in self.values():
            if detail.node:
                detail.node_id = self.classification_tree.node_to_id[detail.node]

    def save(self, path:Path):
        """
        Save the TreeDict to a pickle file.

        Args:
            path (Path): The file path to save the TreeDict.
        """
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)

        self.set_indexes()
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(self, path:Path):
        """
        Load a TreeDict from a pickle file.

        Args:
            path (Path): The path to the serialized TreeDict.

        Returns:
            TreeDict: The loaded TreeDict instance.
        """
        with open(path, 'rb') as handle:
            return pickle.load(handle)

    def node(self, key:str):
        """
        Retrieve the node associated with a key.

        Args:
            key (str): The key for which to retrieve the node.

        Returns:
            SoftmaxNode: The node corresponding to the key.
        """
        detail = self[key]
        if detail.node is not None:
            return detail.node
        return self.classification_tree.node_list[detail.node_id]
    
    def keys_in_partition(self, partition:int):
        """
        Yield all keys that belong to a given partition.

        Args:
            partition (int): The partition to filter by.

        Yields:
            str: Keys in the specified partition.
        """
        for key, detail in self.items():
            if detail.partition == partition:
                yield key

    def keys(self, partition:int|None = None):
        """
        Return keys in the TreeDict, optionally filtering by partition.

        Args:
            partition (int | None): The partition to filter keys by. If None, return all keys.

        Returns:
            Iterator[str]: An iterator over the keys.
        """
        return super().keys() if partition is None else self.keys_in_partition(partition)
    
    def truncate(self, max_depth:int) -> "TreeDict":
        """
        Truncate the classification tree to a specified maximum depth and return a new TreeDict.

        Keys deeper than the depth limit will be reassigned to the ancestor node at that depth.

        Args:
            max_depth (int): The maximum number of ancestor levels to keep.

        Returns:
            TreeDict: A new truncated TreeDict.
        """
        self.classification_tree.set_indexes_if_unset()
        classification_tree = copy.deepcopy(self.classification_tree)
        new_treedict = TreeDict(classification_tree)
        for key in track(self.keys()):
            original_node = self.node(key)
            node_id = self.classification_tree.node_to_id[original_node]
            node = classification_tree.node_list[node_id]

            ancestors = node.ancestors
            if len(ancestors) >= max_depth:
                node = ancestors[max_depth-1]
            new_treedict.add(key, node, self[key].partition)

        # Remove any nodes that beyond the max depth
        for node in new_treedict.classification_tree.pre_order_iter():
            node.readonly = False
            node.softmax_start_index = None
            if len(node.ancestors) >= max_depth:
                node.parent = None

        new_treedict.set_indexes()

        return new_treedict
                
    def add_counts(self):
        """
        Count the number of keys assigned to each node, and store the count in each node.
        """
        for node in self.classification_tree.post_order_iter():
            node.count = 0

        for key in self.keys():
            node = self.node(key)
            node.count += 1

    def add_partition_counts(self):
        """
        Count the number of keys in each partition per node and store it in the node.
        """
        for node in self.classification_tree.post_order_iter():
            node.partition_counts = Counter()

        for key, detail in self.items():
            node = self.node(key)
            partition = detail.partition
            node.partition_counts[partition] += 1

    def render(self, count:bool=False, partition_counts:bool=False, **kwargs):
        """
        Render the tree as text, optionally showing key counts or partition counts.

        Args:
            count (bool): If True, show the number of keys at each node.
            partition_counts (bool): If True, show partition-wise key counts at each node.
            **kwargs: Additional arguments passed to the underlying tree render method.

        Returns:
            anytree.RenderTree or str: The rendered tree.
        """
        if partition_counts:
            self.add_partition_counts()
            for node in self.classification_tree.post_order_iter():
                partition_counts_str = "; ".join([f"{k}->{node.partition_counts[k]}" for k in sorted(node.partition_counts.keys())])
                node.render_str = f"{node.name} {partition_counts_str}"
            kwargs['attr'] = "render_str"
        elif count:
            self.add_counts()
            for node in self.classification_tree.post_order_iter():
                node.render_str = f"{node.name} ({node.count})" if getattr(node, "count", 0) else node.name

            kwargs['attr'] = "render_str"

        return self.classification_tree.render(**kwargs)
    
    def sunburst(self, **kwargs) -> "go.Figure":
        """
        Generate a Plotly sunburst plot based on the TreeDict.

        Node values are based on the number of keys mapped to each node.

        Args:
            **kwargs: Additional keyword arguments passed to Plotly layout.

        Returns:
            plotly.graph_objects.Figure: A sunburst plot.
        """
        import plotly.graph_objects as go

        self.add_counts()
        labels = []
        parents = []
        values = []

        for node in self.classification_tree.pre_order_iter():
            labels.append(node.name)
            parents.append(node.parent.name if node.parent else "")
            values.append(node.count)

        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="remainder",
        ))
        
        fig.update_layout(margin=dict(t=10, l=10, r=10, b=10), **kwargs)
        return fig

    def keys_to_file(self, file:Path) -> None:
        """
        Write all keys to a text file, one per line.

        Args:
            file (Path): Path to the output text file.
        """
        with open(file, "w") as f:
            for key in self.keys():
                print(key, file=f)

    def csv(self, file:Path) -> None:
        """
        Write all keys, node names and partitions to a CSV file.

        Args:
            file (Path): Path to the output text file.
        """
        with open(file, "w") as f:
            print("key,node,partition", file=f)
            for key in self.keys():
                detail = self[key]
                node = self.node(key)
                print(f"{key},{node.name.strip()},{detail.partition}", file=f)

    def pickle_tree(self, output:Path):
        """
        Save only the classification tree (not the key-to-node mapping) to a pickle file.

        Args:
            output (Path): Path to the output file.
        """
        with open(output, 'wb') as pickle_file:
            pickle.dump(self.classification_tree, pickle_file)


app = typer.Typer()

@app.command()
def keys(
    treedict:Path = typer.Argument(...,help="The path to the TreeDict."), 
    partition:int|None = typer.Option(None,help="The index of the partition to list."), 
):
    """ 
    Prints a list of keys in a TreeDict. 
    
    If a partition is given, then only the keys for that partition are given.
    """
    treedict = TreeDict.load(treedict)
    for key in treedict.keys(partition=partition):
        print(key)
    

@app.command()
def csv(
    treedict:Path = typer.Argument(...,help="The path to the TreeDict."), 
    csv:Path = typer.Argument(...,help="The path to the output CSV file."),
):
    """ 
    Writes a CSV file with the key, node name and partition. 
    """
    treedict = TreeDict.load(treedict)
    treedict.csv(csv)
    

@app.command()
def render(
    treedict:Path = typer.Argument(...,help="The path to the TreeDict."), 
    output:Path|None = typer.Option(None, help="The path to save the rendered tree."),
    print_tree:bool = typer.Option(True, help="Whether or not to print the tree to the screen."),
    count:bool = typer.Option(False, help="Whether or not to print the count of keys at each node."),
    partition_counts:bool = typer.Option(False, help="Whether or not to print the count of each partition at each node."),
):
    """
    Render the tree as text, optionally showing key counts or partition counts.
    """
    treedict = TreeDict.load(treedict)
    treedict.render(filepath=output, print=print_tree, count=count, partition_counts=partition_counts)


@app.command()
def count(
    treedict:Path = typer.Argument(...,help="The path to the TreeDict."), 
):
    """
    Prints the number of keys in the TreeDict.
    """
    treedict = TreeDict.load(treedict)
    print(len(treedict))


@app.command()
def sunburst(
    treedict:Path = typer.Argument(...,help="The path to the TreeDict."), 
    show:bool = typer.Option(False, help="Whether or not to show the plot."),
    output:Path = typer.Option(None, help="The path to save the rendered tree."),
    width:int = typer.Option(1000, help="The width of the plot."),
    height:int = typer.Option(0, help="The height of the plot. If 0 then it will be calculated based on the width."),
):
    """
    Renders the TreeDict as a sunburst plot.
    """
    treedict = TreeDict.load(treedict)
    height = height or width

    fig = treedict.sunburst(width=width, height=height)
    if show:
        fig.show()
    
    if output:
        output = Path(output)
        output.parent.mkdir(exist_ok=True, parents=True)

        # if kaleido is installed, turn off mathjax
        # https://github.com/plotly/plotly.py/issues/3469
        try:
            import plotly.io as pio
            pio.kaleido.scope.mathjax = None
        except Exception as e:
            pass

        output_func = fig.write_html if output.suffix.lower() == ".html" else fig.write_image
        output_func(output)


@app.command()
def truncate(
    treedict:Path = typer.Argument(...,help="The path to the TreeDict."), 
    max_depth:int = typer.Argument(...,help="The maximum depth to truncate the tree."),
    output:Path = typer.Argument(...,help="The path to the output file."),
):
    """
    Truncates the tree to a maximum depth.
    """
    treedict = TreeDict.load(treedict)
    new_tree = treedict.truncate(max_depth)
    new_tree.save(output)


@app.command()
def layer_size(
    treedict:Path = typer.Argument(...,help="The path to the TreeDict."),         
):
    """
    Prints the size of the neural network layer to predict the classification tree.
    """    
    treedict = TreeDict.load(treedict)
    print(treedict.classification_tree.layer_size)


@app.command()
def pickle_tree(
    treedict:Path = typer.Argument(...,help="The path to the TreeDict."),    
    output:Path = typer.Argument(...,help="The path to the output pickle file."),     
):
    """
    Pickles the classification tree to a file.
    """
    treedict = TreeDict.load(treedict)
    treedict.pickle_tree(output)
