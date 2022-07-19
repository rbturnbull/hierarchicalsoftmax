import torch

def greedy_predictions(root, all_predictions):
    prediction_nodes = []

    if root.softmax_start_index is None:
        root.set_indexes()

    for predictions in all_predictions:
        node = root
        while (node.children):
            prediction_chlid_index = torch.argmax(predictions[node.softmax_start_index:node.softmax_end_index])
            node = node.children[prediction_chlid_index]

        prediction_nodes.append(node)
    
    return prediction_nodes
