from typing import List, Optional
import torch
from anytree.exporter import DotExporter
from pathlib import Path

from . import nodes


class ShapeError(RuntimeError):
    """
    Raised when the shape of a tensor is different to what is expected.
    """

def full_softmax(prediction_tensor:torch.Tensor, root:nodes.SoftmaxNode) -> torch.Tensor:
    """
    """
    probabilities = torch.zeros_like(prediction_tensor)

    if root.softmax_start_index is None:
        raise nodes.IndexNotSetError(f"The index of the root node {root} has not been set. Call `set_indexes` on this object.")

    if prediction_tensor.shape[-1] != root.layer_size:
        raise ShapeError(
            f"The predictions tensor given to {__name__} has final dimensions of {prediction_tensor.shape[-1]}. "
            "That is not compatible with the root node which expects prediciton tensors to have a final dimension of {root.layer_size}."
        )

    for node in root.node_list:
        if node.is_leaf:
            continue
        elif node == root:
            my_probability = 1.0
        else :
            my_probability = probabilities[:,node.index_in_softmax_layer]
            my_probability = my_probability[:,None]
        
        softmax_probabilities = torch.softmax(
            prediction_tensor[:,node.softmax_start_index:node.softmax_end_index], 
            dim=1,
        )
        probabilities[:,node.softmax_start_index:node.softmax_end_index] = softmax_probabilities * my_probability
    
    return probabilities


def node_probabilities(prediction_tensor:torch.Tensor, root:nodes.SoftmaxNode) -> torch.Tensor:
    probabilities = full_softmax(prediction_tensor, root=root)
    return torch.index_select(probabilities, 1, root.node_indexes_in_softmax_layer)


def leaf_probabilities(prediction_tensor:torch.Tensor, root:nodes.SoftmaxNode) -> torch.Tensor:
    """
    """
    probabilities = full_softmax(prediction_tensor, root=root)
    return torch.index_select(probabilities, 1, root.leaf_indexes_in_softmax_layer)



def greedy_predictions(prediction_tensor:torch.Tensor, root:nodes.SoftmaxNode, max_depth:Optional[int]=None) -> List[nodes.SoftmaxNode]:
    """
    Takes the prediction scores for a number of samples and converts it to a list of predictions of nodes in the tree.

    Predictions use the `greedy` method which means that it chooses the greatest prediction score at each level of the tree.

    Args:
        prediction_tensor (torch.Tensor): The output activations from the softmax layer. Shape (samples, root.layer_size)
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


def greedy_prediction_node_ids(prediction_tensor:torch.Tensor, root:nodes.SoftmaxNode, max_depth:Optional[int]=None) -> List[int]:
    """
    Takes the prediction scores for a number of samples and converts it to a list of predictions of nodes in the tree.

    Predictions use the `greedy` method which means that it chooses the greatest prediction score at each level of the tree.

    Args:
        root (SoftmaxNode): The root softmax node. Needs `set_indexes` to have been called.
        prediction_tensor (torch.Tensor): The predictions coming from the softmax layer. Shape (samples, root.layer_size)
        max_depth (int, optional): If set, then it only gives predictions at a maximum of this number of levels from the root.

    Returns:
        List[int]: A list of node IDs predicted for each sample.
    """
    prediction_nodes = greedy_predictions(prediction_tensor=prediction_tensor, root=root, max_depth=max_depth)
    return root.get_node_ids(prediction_nodes)


def greedy_prediction_node_ids_tensor(prediction_tensor:torch.Tensor, root:nodes.SoftmaxNode, max_depth:Optional[int]=None) -> torch.Tensor:
    node_ids = greedy_prediction_node_ids(prediction_tensor=prediction_tensor, root=root, max_depth=max_depth)
    return torch.as_tensor( node_ids, dtype=int)


def render_probabilities(
        prediction_tensor:torch.Tensor, 
        root:nodes.SoftmaxNode, 
        filepaths:List[Path]=None, 
        prediction_color="red", 
        non_prediction_color="gray",
    ) -> List[DotExporter]:
    """
    Renders the probabilities of each node in the tree as a graphviz graph.

    See https://anytree.readthedocs.io/en/latest/_modules/anytree/exporter/dotexporter.html for more information.

    Args:
        prediction_tensor (torch.Tensor): The output activations from the softmax layer. Shape (samples, root.layer_size)
        root (SoftmaxNode): The root softmax node. Needs `set_indexes` to have been called.
        filepaths (List[Path], optional): Paths to locations where the files can be saved. 
            Can have extension .dot or another format which can be interpreted by GraphViz such as .png or .svg. 
            Defaults to None so that files are not saved.
        prediction_color (str, optional): The color for the greedy prediction nodes and edges. Defaults to "red".
        non_prediction_color (str, optional): The color for the edges which weren't predicted. Defaults to "gray".

    Returns:
        List[DotExporter]: The list of rendered graphs.
    """
    probabilities = full_softmax(prediction_tensor, root=root)
    predictions = greedy_predictions(prediction_tensor, root=root)
    graphs = []
    for my_probabilities, my_prediction in zip(probabilities, predictions):
        greedy_nodes = my_prediction.ancestors + (my_prediction,)
        def node_attributes(node:nodes.SoftmaxNode):
            return f"color={prediction_color}" if node in greedy_nodes else ""
        
        def edge_attributes(parent:nodes.SoftmaxNode, child:nodes.SoftmaxNode):
            color = prediction_color if child in greedy_nodes else non_prediction_color
            return f"label={my_probabilities[child.index_in_softmax_layer]:.2f},color={color}"
        
        graphs.append(DotExporter(root, nodeattrfunc=node_attributes, edgeattrfunc=edge_attributes))

    if filepaths:
        for graph, filepath in zip(graphs, filepaths):
            filepath = Path(filepath)
            filepath.parent.mkdir(exist_ok=True, parents=True)
            if filepath.suffix == ".dot":
                graph.to_dotfile(str(filepath))
            else:
                graph.to_picture(str(filepath))

    return graphs

