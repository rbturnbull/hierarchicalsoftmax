from sklearn.metrics import f1_score
import torch
from . import inference


def greedy_accuracy(prediction_tensor, target_tensor, root):
    prediction_nodes = inference.greedy_predictions(prediction_tensor=prediction_tensor, root=root)
    prediction_node_ids = root.get_node_ids_tensor(prediction_nodes)

    return (prediction_node_ids == target_tensor).float().mean()


def greedy_f1_score(prediction_tensor, target_tensor, root, average="macro"):
    """
    Gives the f1 score of predicting the target.
    """
    prediction_nodes = inference.greedy_predictions(prediction_tensor=prediction_tensor, root=root)
    prediction_node_ids = root.get_node_ids_tensor(prediction_nodes)

    return f1_score(target_tensor.cpu(), prediction_node_ids.cpu(), average=average)

