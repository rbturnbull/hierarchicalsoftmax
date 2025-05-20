from __future__ import annotations
from anytree.exporter import DotExporter
from typing import Union
from pathlib import Path
import torch
from graphviz import Source
from anytree import Node, RenderTree, PreOrderIter, PostOrderIter, LevelOrderIter, LevelOrderGroupIter, ZigZagGroupIter
from typing import List, Optional
from rich.console import Console
console = Console()


class ReadOnlyError(RuntimeError):
    """
    Raised when trying to edit a SoftmaxNode tree after it has been set to read only.
    """


class IndexNotSetError(RuntimeError):
    """
    Raised when set_indexes not set for the SoftmaxNode root.
    """


class AlreadyIndexedError(RuntimeError):
    """
    Raised when set_indexes run more than once on a node.
    """


class SoftmaxNode(Node):
    """
    Creates a hierarchical tree to perform a softmax at each level.
    """
    def __init__(self, *args, alpha:float=1.0, weight=None, label_smoothing:float=0.0, gamma:float = None, readonly:bool=False, **kwargs):
        self.softmax_start_index = None
        self.softmax_end_index = None
        self.children_softmax_end_index = None
        self.node_to_id = None
        self.node_list = None
        self.alpha = alpha
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.readonly = readonly
        self.gamma = gamma # for Focal Loss
        self.children_dict = dict()
        super().__init__(*args, **kwargs)

    def __str__(self):
        return self.name
        
    def __repr__(self):
        return str(self)
        
    def set_indexes(self, index_in_parent:Optional[int]=None, current_index:int=0) -> int:
        """
        Sets all the indexes for this node and its descendants so that each node can be referenced by the root.

        This should be called without arguments only on the root of a hierarchy tree.
        After calling this function the tree from the root down will be read only.

        Args:
            index_in_parent (int, optional): The index of this node in the parent's list of children. 
                Defaults to None which is appropriate for the root of a tree.
            current_index (int, optional): An index value for the root node to reference this node. 
                Defaults to 0 which is appropriate for the root of a tree.

        Returns:
            int: Returns the current_index
        """
        if self.softmax_start_index is not None:
            raise AlreadyIndexedError(f"Node {self} already has been indexed. It cannot be indexed again.")

        self.index_in_parent = index_in_parent
        self.index_in_parent_tensor = torch.as_tensor([index_in_parent], dtype=torch.long) if index_in_parent is not None else None
        
        self.index_in_softmax_layer = self.index_in_parent
        if self.parent:           
            # If the parent has just one child, then this node is skipped in the softmax layer because it isn't needed
            if len(self.parent.children) == 1:
                self.index_in_softmax_layer = None
            else:
                self.index_in_softmax_layer += self.parent.softmax_start_index

        if self.children:
            self.softmax_start_index = current_index
            current_index += len(self.children) if len(self.children) > 1 else 0
            self.softmax_end_index = current_index

            for child_index, child in enumerate(self.children):
                current_index = child.set_indexes(child_index, current_index)

            self.children_softmax_end_index = current_index
        
        # If this is the root, then traverse the tree and make an index of all children
        if self.parent is None:
            self.node_list = [None] * len(self.descendants)
            self.node_to_id = dict()
            non_softmax_index = self.children_softmax_end_index
            for node in self.descendants:
                if node.index_in_softmax_layer is None:
                    node_id = non_softmax_index
                    non_softmax_index += 1
                else:
                    node_id = node.index_in_softmax_layer
                
                self.node_to_id[node] = node_id
                self.node_list[node_id] = node
            
            self.node_list_softmax = self.node_list[:self.children_softmax_end_index] if self.children_softmax_end_index < len(self.node_list) else self.node_list
            self.leaf_list_softmax = [node for node in self.node_list_softmax if not node.children]
            self.node_indexes_in_softmax_layer = torch.as_tensor([node.index_in_softmax_layer for node in self.node_list_softmax])
            self.leaf_indexes = [leaf.best_index_in_softmax_layer() for leaf in self.leaves]
            try:
                self.leaf_indexes = torch.as_tensor(self.leaf_indexes, dtype=torch.long)
            except TypeError:
                pass
        
        self.readonly = True
        return current_index

    def best_index_in_softmax_layer(self) -> int|None:
        if self.index_in_softmax_layer is not None:
            return self.index_in_softmax_layer

        if self.parent:
            return self.parent.best_index_in_softmax_layer()
        
        return None

    def set_indexes_if_unset(self) -> None:
        """ 
        Calls set_indexes if it has not been called yet.
        
        This is only appropriate for the root node.
        """
        if self.root.softmax_start_index is None:
            self.root.set_indexes()

    def render(self, attr:Optional[str]=None, print:bool=False, filepath:Union[str, Path, None] = None, **kwargs) -> RenderTree:
        """
        Renders this node and all its descendants in a tree format.

        Args:
            attr (str, optional): An attribute to print for this rendering of the tree. If None, then the name of each node is used.
            print (bool): Whether or not the tree should be printed. Defaults to False.
            filepath: (str, Path, optional): A path to save the tree to using graphviz. Requires graphviz to be installed.

        Returns:
            RenderTree: The tree rendered by anytree.
        """
        rendered = RenderTree(self, **kwargs)
        if attr:
            rendered = rendered.by_attr(attr)
        if print:
            console.print(rendered)

        if filepath:
            filepath = Path(filepath)
            filepath.parent.mkdir(exist_ok=True, parents=True)

            rendered_tree_graph = DotExporter(self)
            
            if filepath.suffix == ".txt":
                filepath.write_text(str(rendered))
            elif filepath.suffix == ".dot":
                rendered_tree_graph.to_dotfile(str(filepath))
            else:
                rendered_tree_graph.to_picture(str(filepath))

        return rendered
    
    def graphviz(
        self,
        options=None,
        horizontal:bool=True,
    ) -> Source:
        """
        Renders this node and all its descendants in a tree format using graphviz.
        """
        options = options or []
        if horizontal:
            options.append('rankdir="LR";')

        dot_string = "\n".join(DotExporter(self, options=options))

        return Source(dot_string)

    def svg(
        self,
        options=None,
        horizontal:bool=True,
    ) -> str:
        """
        Renders this node and all its descendants in a tree format using graphviz.
        """
        source = self.graphviz(options=options, horizontal=horizontal)
        return source.pipe(format="svg").decode("utf-8")

    def _pre_attach(self, parent:Node):
        if self.readonly or parent.readonly:
            raise ReadOnlyError()

    def _pre_detach(self, parent:Node):
        if self.readonly or parent.readonly:
            raise ReadOnlyError()

    def _post_attach(self, parent:Node):
        """Method call after attaching to `parent`."""
        parent.children_dict[self.name] = self

    def _post_detach(self, parent:Node):
        """Method call after detaching from `parent`."""
        del parent.children_dict[self.name]

    def get_child_by_name(self, name:str) -> SoftmaxNode:
        """
        Returns the child node that has the same name as what is given.

        Args:
            name (str): The name of the child node requested.

        Returns:
            SoftmaxNode: The child node that has the same name as what is given. If not child node exists with this name then `None` is returned.
        """
        return self.children_dict.get(name, None)

    def get_node_ids(self, nodes:List) -> List[int]:
        """
        Gets the index values for descendant nodes.

        This should only be used for root nodes. 
        If `set_indexes` has been yet called on this object then it is performed as part of this function call.

        Args:
            nodes (List): A list of descendant nodes.

        Returns:
            List[int]: A list of indexes for the descendant nodes requested.
        """
        if self.node_to_id is None:
            self.set_indexes()

        return [self.node_to_id[node] for node in nodes]

    def get_node_ids_tensor(self, nodes:List) -> torch.Tensor:
        """
        Gets the index values for descendant nodes.

        This should only be used for root nodes. 
        If `set_indexes` has been yet called on this object then it is performed as part of this function call.

        Args:
            nodes (List): A list of descendant nodes.

        Returns:
            torch.Tensor: A tensor which contains the indexes for the descendant nodes requested.
        """
        return torch.as_tensor( self.get_node_ids(nodes), dtype=int)
    
    @property
    def layer_size(self) -> int:
        self.root.set_indexes_if_unset()

        return self.children_softmax_end_index
    
    def render_equal(self, string_representation:str, **kwargs) -> bool:
        """
        Checks if the string representation of this node and its descendants matches the given string.

        Args:
            string_representation (str): The string representation to compare to.
        """
        my_render = str(self.render(**kwargs))
        lines1 = str(my_render).strip().split("\n")
        lines2 = str(string_representation).strip().split("\n")

        if len(lines1) != len(lines2):
            return False

        for line1, line2 in zip(lines1, lines2):
            if line1.strip() != line2.strip():
                return False
            
        return True

    def pre_order_iter(self, depth=None, **kwargs) -> PreOrderIter:
        """ 
        Returns a pre-order iterator.
        
        See https://anytree.readthedocs.io/en/latest/api/anytree.iterators.html#anytree.iterators.preorderiter.PreOrderIter
        """
        if depth is not None:
            kwargs["maxlevel"] = depth + 1
        return PreOrderIter(self, **kwargs)
    
    def post_order_iter(self, depth=None, **kwargs) -> PostOrderIter:
        """ 
        Returns a post-order iterator.
        
        See https://anytree.readthedocs.io/en/latest/api/anytree.iterators.html#anytree.iterators.postorderiter.PostOrderIter
        """
        if depth is not None:
            kwargs["maxlevel"] = depth + 1
        return PostOrderIter(self, **kwargs)

    def level_order_iter(self, depth=None, **kwargs) -> LevelOrderIter:
        """ 
        Returns a level-order iterator.
        
        See https://anytree.readthedocs.io/en/latest/api/anytree.iterators.html#anytree.iterators.levelorderiter.LevelOrderIter
        """
        if depth is not None:
            kwargs["maxlevel"] = depth + 1
        return LevelOrderIter(self, **kwargs)

    def level_order_group_iter(self, depth=None, **kwargs) -> LevelOrderGroupIter:
        """ 
        Returns a level-order iterator with grouping starting at this node.
        
        https://anytree.readthedocs.io/en/latest/api/anytree.iterators.html#anytree.iterators.levelordergroupiter.LevelOrderGroupIter
        """
        if depth is not None:
            kwargs["maxlevel"] = depth + 1
        return LevelOrderGroupIter(self, **kwargs)

    def zig_zag_group_iter(self, depth=None, **kwargs) -> ZigZagGroupIter:
        """ 
        Returns a zig-zag iterator with grouping starting at this node.
        
        https://anytree.readthedocs.io/en/latest/api/anytree.iterators.html#anytree.iterators.zigzaggroupiter.ZigZagGroupIter
        """
        if depth is not None:
            kwargs["maxlevel"] = depth + 1
        return ZigZagGroupIter(self, **kwargs)

