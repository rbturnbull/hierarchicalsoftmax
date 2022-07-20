from sklearn.metrics import f1_score
import torch
from . import inference, nodes


def greedy_accuracy(prediction_tensor, target_tensor, root):
    """
    Gives the accuracy of predicting the target in a hierarchy tree.

    Predictions use the `greedy` method which means that it chooses the greatest prediction score at each level of the tree.

    Args:
        prediction_tensor (torch.Tensor): A tensor with the raw scores for each node in the tree. Shape: (samples, root.children_softmax_end_index)
        target_tensor (torch.Tensor): A tensor with the target node indexes. Shape: (samples,).
        root (SoftmaxNode): The root of the hierarchy tree.

    Returns:
        float: The accuracy value (i.e. the number that are correct divided by the total number of samples)
    """    
    prediction_nodes = inference.greedy_predictions(prediction_tensor=prediction_tensor, root=root)
    prediction_node_ids = root.get_node_ids_tensor(prediction_nodes)

    return (prediction_node_ids == target_tensor).float().mean()


def greedy_f1_score(prediction_tensor:torch.Tensor, target_tensor:torch.Tensor, root:nodes.SoftmaxNode, average:str="macro") -> float:
    """
    Gives the f1 score of predicting the target i.e. a harmonic mean of the precision and recall.

    Predictions use the `greedy` method which means that it chooses the greatest prediction score at each level of the tree.

    Args:
        prediction_tensor (torch.Tensor): A tensor with the raw scores for each node in the tree. Shape: (samples, root.children_softmax_end_index)
        target_tensor (torch.Tensor): A tensor with the target node indexes. Shape: (samples,).
        root (SoftmaxNode): The root of the hierarchy tree.
        average (str, optional): The type of averaging over the different classes.
            Options are: 'micro', 'macro', 'samples', 'weighted', 'binary' or None. 
            See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html for details.
            Defaults to "macro".

    Returns:
        float: The f1 score 
    """
    prediction_nodes = inference.greedy_predictions(prediction_tensor=prediction_tensor, root=root)
    prediction_node_ids = root.get_node_ids_tensor(prediction_nodes)

    return f1_score(target_tensor.cpu(), prediction_node_ids.cpu(), average=average)

