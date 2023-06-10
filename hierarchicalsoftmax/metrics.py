from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from . import inference, nodes


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


