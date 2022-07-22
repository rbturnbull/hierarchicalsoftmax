import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class HierarchicalSoftmaxLoss(nn.Module):
    """
    A module which sums the loss for each level of a hiearchical tree.
    """
    def __init__(
        self,
        root,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.root = root

        # Set the indexes of the tree if necessary
        self.root.set_indexes_if_unset()

        assert len(self.root.node_list) > 0

    def forward(self, batch_predictions: Tensor, targets: Tensor) -> Tensor:
        target_nodes = (self.root.node_list[target] for target in targets)

        loss = 0.0
        device = targets.device

        for prediction, target_node in zip(batch_predictions, target_nodes):
            node = target_node
            while node.parent:
                node.index_in_parent_tensor = node.index_in_parent_tensor.to(device) # can this be done elsewhere?
                loss += node.parent.alpha * F.cross_entropy(
                    torch.unsqueeze(prediction[node.parent.softmax_start_index:node.parent.softmax_end_index], dim=0),
                    node.index_in_parent_tensor,
                    weight=node.parent.weight,
                    label_smoothing=node.parent.label_smoothing,
                )
                node = node.parent
                
        batch_size = len(targets)
        loss /= batch_size
        return loss
