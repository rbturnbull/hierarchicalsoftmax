from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from typing import Callable
from collections.abc import Sequence
from torch.nn import Module
from torch import Tensor
from torchmetrics.metric import Metric, apply_to_collection

from . import inference, nodes
from .inference import ShapeError



def target_max_depth(target_tensor:torch.Tensor, root:nodes.SoftmaxNode, max_depth:int):
    """ Converts the target tensor to the max depth of the tree. """
    if max_depth:
        max_depth_target_nodes = [root.node_list[target].path[:max_depth+1][-1] for target in target_tensor]
        target_tensor = root.get_node_ids_tensor(max_depth_target_nodes)

    return target_tensor


def greedy_accuracy(prediction_tensor, target_tensor, root, max_depth=None):
    """
    Gives the accuracy of predicting the target in a hierarchy tree.

    Predictions use the `greedy` method which means that it chooses the greatest prediction score at each level of the tree.

    Args:
        prediction_tensor (torch.Tensor): A tensor with the raw scores for each node in the tree. Shape: (samples, root.layer_size)
        target_tensor (torch.Tensor): A tensor with the target node indexes. Shape: (samples,).
        root (SoftmaxNode): The root of the hierarchy tree.

    Returns:
        float: The accuracy value (i.e. the number that are correct divided by the total number of samples)
    """    
    prediction_node_ids = inference.greedy_prediction_node_ids_tensor(prediction_tensor=prediction_tensor, root=root, max_depth=max_depth)
    target_tensor = target_max_depth(target_tensor, root, max_depth)

    return (prediction_node_ids.to(target_tensor.device) == target_tensor).float().mean()


def greedy_accuracy_depth_one(prediction_tensor, target_tensor, root):
    return greedy_accuracy(prediction_tensor, target_tensor, root, max_depth=1)
    

def greedy_accuracy_depth_two(prediction_tensor, target_tensor, root):
    return greedy_accuracy(prediction_tensor, target_tensor, root, max_depth=2)


def depth_accurate(prediction_tensor, target_tensor, root:nodes.SoftmaxNode, max_depth:int=0, threshold:float|None=None):
    """ Returns a tensor of shape (samples,) with the depth of predictions which were accurate """
    depths = []

    if root.softmax_start_index is None:
        raise nodes.IndexNotSetError(f"The index of the root node {root} has not been set. Call `set_indexes` on this object.")

    if isinstance(prediction_tensor, tuple) and len(prediction_tensor) == 1:
        prediction_tensor = prediction_tensor[0]

    if prediction_tensor.shape[-1] != root.layer_size:
        raise ShapeError(
            f"The predictions tensor given to {__name__} has final dimensions of {prediction_tensor.shape[-1]}. "
            f"That is not compatible with the root node which expects prediciton tensors to have a final dimension of {root.layer_size}."
        )

    for predictions, target in zip(prediction_tensor, target_tensor):
        node = root
        depth = 0
        target_node = root.node_list[target]
        target_path = target_node.path
        target_path_length = len(target_path)


        while (node.children):
            # This would be better if we could use torch.argmax but it doesn't work with MPS in the production version of pytorch
            # See https://github.com/pytorch/pytorch/issues/98191
            # https://github.com/pytorch/pytorch/pull/104374
            if len(node.children) == 1:
                # if this node use just one child, then we don't check the prediction
                prediction_child_index = 0
            else:
                prediction_child_index = torch.max(predictions[node.softmax_start_index:node.softmax_end_index], dim=0).indices

            node = node.children[prediction_child_index]
            depth += 1

            if depth < target_path_length and node != target_path[depth]:
                depth -= 1
                break
            
            # Stop if we have reached the maximum depth
            if max_depth and depth >= max_depth:
                break

        depths.append(depth)
    
    return torch.tensor(depths, dtype=int)


def greedy_f1_score(prediction_tensor:torch.Tensor, target_tensor:torch.Tensor, root:nodes.SoftmaxNode, average:str="macro", max_depth=None) -> float:
    """
    Gives the f1 score of predicting the target i.e. a harmonic mean of the precision and recall.

    Predictions use the `greedy` method which means that it chooses the greatest prediction score at each level of the tree.

    Args:
        prediction_tensor (torch.Tensor): A tensor with the raw scores for each node in the tree. Shape: (samples, root.layer_size)
        target_tensor (torch.Tensor): A tensor with the target node indexes. Shape: (samples,).
        root (SoftmaxNode): The root of the hierarchy tree.
        average (str, optional): The type of averaging over the different classes.
            Options are: 'micro', 'macro', 'samples', 'weighted', 'binary' or None. 
            See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html for details.
            Defaults to "macro".

    Returns:
        float: The f1 score 
    """
    prediction_node_ids = inference.greedy_prediction_node_ids_tensor(prediction_tensor=prediction_tensor, root=root, max_depth=max_depth)
    target_tensor = target_max_depth(target_tensor, root, max_depth)

    return f1_score(target_tensor.cpu(), prediction_node_ids.cpu(), average=average)


def greedy_precision(prediction_tensor:torch.Tensor, target_tensor:torch.Tensor, root:nodes.SoftmaxNode, average:str="macro", max_depth=None) -> float:
    """
    Gives the precision score of predicting the target.

    Predictions use the `greedy` method which means that it chooses the greatest prediction score at each level of the tree.

    Args:
        prediction_tensor (torch.Tensor): A tensor with the raw scores for each node in the tree. Shape: (samples, root.layer_size)
        target_tensor (torch.Tensor): A tensor with the target node indexes. Shape: (samples,).
        root (SoftmaxNode): The root of the hierarchy tree.
        average (str, optional): The type of averaging over the different classes.
            Options are: 'micro', 'macro', 'samples', 'weighted', 'binary' or None. 
            See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html for details.
            Defaults to "macro".

    Returns:
        float: The precision 
    """
    prediction_node_ids = inference.greedy_prediction_node_ids_tensor(prediction_tensor=prediction_tensor, root=root, max_depth=max_depth)
    target_tensor = target_max_depth(target_tensor, root, max_depth)
    
    return precision_score(target_tensor.cpu(), prediction_node_ids.cpu(), average=average)


def greedy_recall(prediction_tensor:torch.Tensor, target_tensor:torch.Tensor, root:nodes.SoftmaxNode, average:str="macro", max_depth=None) -> float:
    """
    Gives the recall score of predicting the target.

    Predictions use the `greedy` method which means that it chooses the greatest prediction score at each level of the tree.

    Args:
        prediction_tensor (torch.Tensor): A tensor with the raw scores for each node in the tree. Shape: (samples, root.layer_size)
        target_tensor (torch.Tensor): A tensor with the target node indexes. Shape: (samples,).
        root (SoftmaxNode): The root of the hierarchy tree.
        average (str, optional): The type of averaging over the different classes.
            Options are: 'micro', 'macro', 'samples', 'weighted', 'binary' or None. 
            See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html for details.
            Defaults to "macro".

    Returns:
        float: The recall 
    """
    prediction_node_ids = inference.greedy_prediction_node_ids_tensor(prediction_tensor=prediction_tensor, root=root, max_depth=max_depth)
    target_tensor = target_max_depth(target_tensor, root, max_depth)

    return recall_score(target_tensor.cpu(), prediction_node_ids.cpu(), average=average)


def greedy_accuracy_parent(prediction_tensor, target_tensor, root, max_depth=None):
    """
    Gives the accuracy of predicting the parent of the target in a hierarchy tree.

    Predictions use the `greedy` method which means that it chooses the greatest prediction score at each level of the tree.

    Args:
        prediction_tensor (torch.Tensor): A tensor with the raw scores for each node in the tree. Shape: (samples, root.layer_size)
        target_tensor (torch.Tensor): A tensor with the target node indexes. Shape: (samples,).
        root (SoftmaxNode): The root of the hierarchy tree.

    Returns:
        float: The accuracy value (i.e. the number that are correct divided by the total number of samples)
    """    
    prediction_nodes = inference.greedy_predictions(prediction_tensor=prediction_tensor, root=root, max_depth=max_depth)
    prediction_parents = [node.parent for node in prediction_nodes]
    prediction_parent_ids = root.get_node_ids_tensor(prediction_parents)

    target_tensor = target_max_depth(target_tensor, root, max_depth)
    target_parents = [root.node_list[target].parent for target in target_tensor]
    target_parent_ids = root.get_node_ids_tensor(target_parents)

    return (prediction_parent_ids.to(target_parent_ids.device) == target_parent_ids).float().mean()


class GreedyAccuracy():
    name:str = "greedy"

    def __init__(self, root:nodes.SoftmaxNode, name="greedy_accuracy", max_depth=None):
        self.max_depth = max_depth
        self.name = name
        self.root = root

    @property
    def __name__(self):
        """ For using as a FastAI metric. """
        return self.name
    
    def __call__(self, predictions, targets):
        return greedy_accuracy(predictions, targets, self.root, max_depth=self.max_depth)


class HierarchicalSoftmaxTorchMetric(Metric):
    def _apply(self, fn: Callable, exclude_state: Sequence[str] = "") -> Module:
        """Overwrite `_apply` function such that we can also move metric states to the correct device.

        This method is called by the base ``nn.Module`` class whenever `.to`, `.cuda`, `.float`, `.half` etc. methods
        are called. Dtype conversion is guarded and will only happen through the special `set_dtype` method.

        Overriding because there is an issue device in the parent class.

        Args:
            fn: the function to apply
            exclude_state: list of state variables to exclude from applying the function, that then needs to be handled
                by the metric class itself.
        """
        this = super(Metric, self)._apply(fn)
        fs = str(fn)
        cond = any(f in fs for f in ["Module.type", "Module.half", "Module.float", "Module.double", "Module.bfloat16"])
        if not self._dtype_convert and cond:
            return this

        # Also apply fn to metric states and defaults
        for key, value in this._defaults.items():
            if key in exclude_state:
                continue

            if isinstance(value, Tensor):
                this._defaults[key] = fn(value)
            elif isinstance(value, Sequence):
                this._defaults[key] = [fn(v) for v in value]

            current_val = getattr(this, key)
            if isinstance(current_val, Tensor):
                setattr(this, key, fn(current_val))
            elif isinstance(current_val, Sequence):
                setattr(this, key, [fn(cur_v) for cur_v in current_val])
            else:
                raise TypeError(
                    f"Expected metric state to be either a Tensor or a list of Tensor, but encountered {current_val}"
                )

        # Additional apply to forward cache and computed attributes (may be nested)
        if this._computed is not None:
            this._computed = apply_to_collection(this._computed, Tensor, fn)
        if this._forward_cache is not None:
            this._forward_cache = apply_to_collection(this._forward_cache, Tensor, fn)

        return this


class GreedyAccuracyTorchMetric(HierarchicalSoftmaxTorchMetric):
    def __init__(self, root:nodes.SoftmaxNode, name:str="", max_depth=None):
        super().__init__()
        self.root = root
        self.max_depth = max_depth
        self.name = name or (f"greedy_accuracy_{max_depth}" if max_depth else "greedy_accuracy")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        self.total += targets.size(0)
        self.correct += int(greedy_accuracy(predictions, targets, self.root, max_depth=self.max_depth) * targets.size(0))

    def compute(self):
        return self.correct / self.total
    

class RankAccuracyTorchMetric(HierarchicalSoftmaxTorchMetric):
    def __init__(self, root, ranks: dict[int, str], name: str = "rank_accuracy"):
        super().__init__()
        self.root = root
        self.ranks = ranks
        self.name = name

        # Use `add_state` for metrics to handle distributed reduction and device placement
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        for rank_name in ranks.values():
            self.add_state(rank_name, default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        if isinstance(predictions, tuple) and len(predictions) == 1:
            predictions = predictions[0]

        # Ensure tensors match the device
        predictions = predictions.to(self.device)
        targets = targets.to(self.device)

        self.total += targets.size(0)
        depth_accurate_tensor = depth_accurate(predictions, targets, self.root)

        for depth, rank_name in self.ranks.items():
            accurate_at_depth = (depth_accurate_tensor >= depth).sum()
            setattr(self, rank_name, getattr(self, rank_name) + accurate_at_depth)

    def compute(self):
        # Compute final metric values
        return {
            rank_name: getattr(self, rank_name) / self.total
            for rank_name in self.ranks.values()
        }


class LeafAccuracyTorchMetric(HierarchicalSoftmaxTorchMetric):
    def __init__(self, root:nodes.SoftmaxNode, name:str="", max_depth=None):
        super().__init__()
        self.root = root
        self.max_depth = max_depth
        self.name = name or (f"leaf_accuracy_{max_depth}" if max_depth else "leaf_accuracy")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.node_indexes = torch.as_tensor([node.best_index_in_softmax_layer() for node in self.root.node_list])
        self.leaf_indexes = torch.as_tensor(self.root.leaf_indexes)

    def update(self, predictions, targets):
        self.total += targets.size(0)
        
        # Make sure the tensors are on the same device
        self.node_indexes = self.node_indexes.to(predictions.device)
        self.leaf_indexes = self.leaf_indexes.to(predictions.device)

        target_indices = torch.index_select(self.node_indexes.to(targets.device), 0, targets)
        
        # get indices of the maximum values along the last dimension
        probabilities = inference.leaf_probabilities(prediction_tensor=predictions, root=self.root)
        _, max_indices = torch.max(probabilities, dim=1)
        predicted_leaf_indices = torch.index_select(self.root.leaf_indexes.to(targets.device), 0, max_indices)

        self.correct += (predicted_leaf_indices == target_indices).sum()

    def compute(self):
        return self.correct / self.total
    

