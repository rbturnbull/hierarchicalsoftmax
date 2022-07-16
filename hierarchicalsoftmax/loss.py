from torch import nn
from torch import Tensor
import torch.nn.functional as F


class HierarchicalSoftmaxLoss(nn.Module):
    def __init__(
        self,
        root,
        prediction_nodes, # is this necessary?
        **kwargs
    ):
        super().__init__(**kwargs)
        self.root = root

    def forward(self, batch_predictions: Tensor, targets: Tensor) -> Tensor:
        target_nodes = (self.prediction_nodes[target] for target in targets)

        loss = 0.0
        for prediction, target_node in zip(batch_predictions, target_nodes):
            node = target_node
            while node.parent:
                loss += node.parent.alpha * F.cross_entropy(
                    prediction[node.softmax_start_index:node.softmax_end_index],
                    node.index_in_parent,
                    weight=node.parent.weight,
                    label_smoothing=node.parent.label_smoothing,
                )
                node = node.parent

        batch_size = len(targets)
        loss /= batch_size
        return loss
