from typing import List
import torch

from . import nodes


class ShapeError(RuntimeError):
    pass


def greedy_predictions(root:nodes.SoftmaxNode, prediction_tensor:torch.Tensor) -> List[nodes.SoftmaxNode]:
    """
    Takes the prediction scores for a number of samples and converts it to a list of predictions of nodes in the tree.

    Args:
        root (SoftmaxNode): The root softmax node. Needs `set_indexes` to have been called.
        prediction_tensor (torch.Tensor): The predictions coming from the softmax layer. Size (samples, root.children_softmax_end_index)

    Returns:
        List[nodes.SoftmaxNode]: A list of nodes predicted for each sample.
    """
    prediction_nodes = []

    if root.softmax_start_index is None:
        raise nodes.IndexNotSetError(f"The index of the root node {root} has not been set. Call `set_indexes` on this object.")

    if prediction_tensor.shape[-1] != root.children_softmax_end_index:
        raise ShapeError(
            f"The predictions tensor given to {__func__} has final dimensions of {prediction_tensor.shape[-1]}. "
            "That is not compatible with the root node which expects prediciton tensors to have a final dimension of {root.children_softmax_end_index}."
        )

    for predictions in prediction_tensor:
        node = root
        while (node.children):
            prediction_chlid_index = torch.argmax(predictions[node.softmax_start_index:node.softmax_end_index])
            node = node.children[prediction_chlid_index]

        prediction_nodes.append(node)
    
    return prediction_nodes
