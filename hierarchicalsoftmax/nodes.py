from pathlib import Path
from torch import nn
from anytree import Node, RenderTree

from rich.console import Console
console = Console()


class ReadOnlyError(RuntimeError):
    pass


class SoftmaxNode(Node):
    def __init__(self, *args, alpha=1.0, weight=None, label_smoothing=0.0, readonly=False, **kwargs):
        self.softmax_start_index = None
        self.softmax_end_index = None
        self.children_softmax_end_index = None
        self.alpha = alpha
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.readonly = readonly
        super().__init__(*args, **kwargs)

    def __str__(self):
        return self.name
        
    def __repr__(self):
        return str(self)
        
    def set_indexes(self, index_in_parent=None, current_index=0):
        assert self.softmax_start_index is None
        self.index_in_parent = index_in_parent
        if self.children:
            self.softmax_start_index = current_index
            current_index += len(self.children)
            self.softmax_end_index = current_index

            for child_index, child in enumerate(self.children):
                current_index = child.set_indexes(child_index, current_index)

            self.children_softmax_end_index = current_index
        self.readonly = True
        print('self.readonly', 'set_indexes', self.readonly)
        return current_index

    def render(self, *args, attr=None, print=False, **kwargs):
        rendered = RenderTree(self, *args, **kwargs)
        if attr:
            rendered = rendered.by_attr(attr)
        if print:
            console.print(rendered)
        return rendered

    def _pre_attach(self, parent):
        if self.readonly or parent.readonly:
            raise ReadOnlyError()

    def _pre_detach(self, parent):
        if self.readonly or parent.readonly:
            raise ReadOnlyError()
