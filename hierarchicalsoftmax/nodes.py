from __future__ import annotations
import torch
from anytree import Node, RenderTree, PreOrderIter
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
    def __init__(self, *args, alpha:float=1.0, weight=None, label_smoothing:float=0.0, readonly:bool=False, **kwargs):
        self.softmax_start_index = None
        self.softmax_end_index = None
        self.children_softmax_end_index = None
        self.alpha = alpha
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.readonly = readonly
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
        self.index_in_parent_tensor = torch.as_tensor(index_in_parent, dtype=torch.long) if index_in_parent is not None else None
        if self.children:
            self.softmax_start_index = current_index
            current_index += len(self.children)
            self.softmax_end_index = current_index

            for child_index, child in enumerate(self.children):
                current_index = child.set_indexes(child_index, current_index)

            self.children_softmax_end_index = current_index
        
        # If this is the root, then traverse the tree and make an index of all children
        if self.softmax_start_index == 0:
            self.node_list = list(PreOrderIter(self))
            self.node_to_id = {node:index for index, node in enumerate(self.node_list)}
        
        self.readonly = True
        return current_index

    def render(self, attr:Optional[str]=None, print:bool=False, **kwargs) -> RenderTree:
        """
        Renders this node and all its descendants in a tree format.

        Args:
            attr (str, optional): An attribute to print for this rendering of the tree. If None, then the name of each node is used.
            print (bool): Whether or not the tree should be printed. Defaults to False.

        Returns:
            RenderTree: The tree rendered by anytree.
        """
        rendered = RenderTree(self, **kwargs)
        if attr:
            rendered = rendered.by_attr(attr)
        if print:
            console.print(rendered)
        return rendered

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
        If `set_indexes` has been yet called on this object then it is performed as part of this function.

        Args:
            nodes (List): A list of descendant nodes.

        Returns:
            List[int]: A list of indexes for the descendant nodes requested.
        """
        if self.node_to_id is None:
            self.set_indexes()

        return [self.node_to_id[node] for node in nodes]

    def get_node_ids_tensor(self, nodes:List) -> torch.Tensor:
        return torch.as_tensor( self.get_node_ids(nodes), dtype=int )