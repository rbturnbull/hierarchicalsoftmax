import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable


def focal_loss_with_smoothing(logits, label, weight=None, gamma=0.0, label_smoothing=0.0):
    """ 
    Adapted from https://github.com/Kageshimasu/focal-loss-with-smoothing 
    and https://github.com/clcarwin/focal_loss_pytorch
    """
    log_probabilities = F.log_softmax(logits, dim=-1)
    label = label.view(-1,1)
    log_probability = log_probabilities.gather(1,label).squeeze()
    n_classes = logits.size(1)
    uniform_probability = label_smoothing / n_classes
    label_distribution = torch.full_like(logits, uniform_probability)
    label_distribution.scatter_(1, label, 1.0 - label_smoothing + uniform_probability)

    probability = Variable(log_probability.data.exp())
    difficulty_level = (1-probability)** gamma
    loss = -difficulty_level * torch.sum(log_probabilities * label_distribution, dim=1)

    # Weights
    if weight is not None:
        weight = weight.to(logits.device)
        loss *= torch.gather(weight, -1, label.squeeze())/weight.mean()

    return loss.mean()


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

        total_loss = 0.0
        device = targets.device

        for prediction, target_node in zip(batch_predictions, target_nodes):
            node = target_node
            while node.parent:
                # if this is the sole child, then skip it
                if len(node.parent.children) == 1:
                    node = node.parent
                    continue
                
                node.index_in_parent_tensor = node.index_in_parent_tensor.to(device) # can this be done elsewhere?
                logits = torch.unsqueeze(prediction[node.parent.softmax_start_index:node.parent.softmax_end_index], dim=0)
                label = node.index_in_parent_tensor                
                weight = node.parent.weight
                label_smoothing = node.parent.label_smoothing
                gamma = node.parent.gamma
                if gamma is not None and gamma > 0.0:
                    loss = focal_loss_with_smoothing(
                        logits,
                        label,
                        weight=weight,
                        label_smoothing=label_smoothing,
                        gamma=gamma,
                    )
                else:
                    loss = F.cross_entropy(
                        logits,
                        label,
                        weight=weight,
                        label_smoothing=label_smoothing,
                    )

                total_loss += node.parent.alpha * loss
                node = node.parent
                
        batch_size = len(targets)
        total_loss /= batch_size
        return total_loss




# class HierarchicalSoftmaxLoss(nn.Module):
#     """
#     A module which sums the loss for each level of a hiearchical tree.
#     """
#     def __init__(
#         self,
#         root,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.root = root

#         # Set the indexes of the tree if necessary
#         self.root.set_indexes_if_unset()

#         assert len(self.root.node_list) > 0

#     def forward(self, batch_predictions: Tensor, targets: Tensor) -> Tensor:
#         target_nodes = (self.root.node_list[target] for target in targets)

#         loss = 0.0
#         device = targets.device

#         for prediction, target_node in zip(batch_predictions, target_nodes):
#             node = target_node
#             while node.parent:
#                 node.index_in_parent_tensor = node.index_in_parent_tensor.to(device) # can this be done elsewhere?
#                 print(node.index_in_parent_tensor)
#                 loss += node.parent.alpha * F.cross_entropy(
#                     torch.unsqueeze(prediction[node.parent.softmax_start_index:node.parent.softmax_end_index], dim=0),
#                     node.index_in_parent_tensor,
#                     weight=node.parent.weight,
#                     label_smoothing=node.parent.label_smoothing,
#                 )
#                 node = node.parent
                
#         batch_size = len(targets)
#         loss /= batch_size
#         return loss