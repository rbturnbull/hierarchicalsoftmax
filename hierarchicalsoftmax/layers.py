from torch import nn

from .nodes import SoftmaxNode
from .tensors import LazyLinearTensor

class HierarchicalSoftmaxLayerError(RuntimeError):
    pass

class HierarchicalSoftmaxLayerMixin():
    def __init__(self, root:SoftmaxNode, out_features=None, **kwargs):
        self.root = root

        if out_features is not None:
            raise HierarchicalSoftmaxLayerError(
                "Trying to create a HierarchicalSoftmaxLinearLayer by explicitly setting `out_features`. "
                "This value should be determined from the hierarchy tree and not `out_features` argument should be given to HierarchicalSoftmaxLinearLayer."
            )

        super().__init__(out_features=self.root.layer_size, **kwargs)

    def forward(self, x) -> LazyLinearTensor:
        return LazyLinearTensor(x, weight=self.weight, bias=self.bias)
    

class HierarchicalSoftmaxLinear(HierarchicalSoftmaxLayerMixin, nn.Linear):
    """
    Creates a linear layer designed to be the final layer in a neural network model that produces unnormalized scores given to HierarchicalSoftmaxLoss.

    The `out_features` value is set internally from root.layer_size and cannot be given as an argument.
    """


class HierarchicalSoftmaxLazyLinear(HierarchicalSoftmaxLayerMixin, nn.LazyLinear):
    """
    Creates a lazy linear layer designed to be the final layer in a neural network model that produces unnormalized scores given to HierarchicalSoftmaxLoss.

    The `out_features` value is set internally from root.layer_size and cannot be given as an argument.
    The `in_features` will be inferred from the previous layer at runtime.
    """
