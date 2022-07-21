from typing import List, Optional
import torch

from . import nodes


class ShapeError(RuntimeError):
    """
    Raised when the shape of a tensor is different to what is expected.
    """


def greedy_predictions(prediction_tensor:torch.Tensor, root:nodes.SoftmaxNode, max_depth:Optional[int]=None) -> List[nodes.SoftmaxNode]:
    """
    Takes the prediction scores for a number of samples and converts it to a list of predictions of nodes in the tree.

    Predictions use the `greedy` method which means that it chooses the greatest prediction score at each level of the tree.

    Args:
        root (SoftmaxNode): The root softmax node. Needs `set_indexes` to have been called.
        prediction_tensor (torch.Tensor): The predictions coming from the softmax layer. Shape (samples, root.layer_size)
        max_depth (int, optional): If set, then it only gives predictions at a maximum of this number of levels from the root.

    Returns:
        List[nodes.SoftmaxNode]: A list of nodes predicted for each sample.
    """
    prediction_nodes = []

    if root.softmax_start_index is None:
        raise nodes.IndexNotSetError(f"The index of the root node {root} has not been set. Call `set_indexes` on this object.")

    if prediction_tensor.shape[-1] != root.layer_size:
        raise ShapeError(
            f"The predictions tensor given to {__name__} has final dimensions of {prediction_tensor.shape[-1]}. "
            "That is not compatible with the root node which expects prediciton tensors to have a final dimension of {root.layer_size}."
        )

    for predictions in prediction_tensor:
        node = root
        depth = 1
        while (node.children):
            prediction_chlid_index = torch.argmax(predictions[node.softmax_start_index:node.softmax_end_index])
            node = node.children[prediction_chlid_index]

            # Stop if we have reached the maximum depth
            if max_depth and depth >= max_depth:
                break
            depth += 1

        prediction_nodes.append(node)
    
    return prediction_nodes
