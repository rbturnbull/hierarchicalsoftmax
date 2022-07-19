import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class HierarchicalSoftmaxLoss(nn.Module):
    def __init__(
        self,
        root,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.root = root

        # Set the indexes of the tree if necessary
        if self.root.softmax_start_index is None:
            self.root.set_indexes()
        assert len(self.root.node_list) > 0

    def forward(self, batch_predictions: Tensor, targets: Tensor) -> Tensor:
        target_nodes = (self.root.node_list[target] for target in targets)

        loss = 0.0
        for prediction, target_node in zip(batch_predictions, target_nodes):
            node = target_node
            while node.parent:
                print('loss', loss)
                print('node', node)
                loss += node.parent.alpha * F.cross_entropy(
                    prediction[node.parent.softmax_start_index:node.parent.softmax_end_index],
                    torch.as_tensor(node.index_in_parent, dtype=torch.long), # should be created once
                    weight=node.parent.weight,
                    label_smoothing=node.parent.label_smoothing,
                )
                print('loss after x entropy', loss)
                node = node.parent
                

        batch_size = len(targets)
        loss /= batch_size
        return loss