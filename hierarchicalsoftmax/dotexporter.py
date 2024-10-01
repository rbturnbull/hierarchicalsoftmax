from anytree.exporter import DotExporter
from anytree import PreOrderIter

class ThresholdDotExporter(DotExporter):
    def __init__(
        self, 
        node, 
        probabilities, 
        greedy_nodes, 
        graph="digraph", 
        name="tree", 
        options=None,
        indent=4, 
        nodenamefunc=None, 
        nodeattrfunc=None,
        edgeattrfunc=None, 
        edgetypefunc=None,
        prediction_color="red",
        non_prediction_color="gray",
        horizontal:bool=True,
        threshold:float=0.005,
    ):
        options = options or []
        if horizontal:
            options.append('rankdir="LR";')
        
        super().__init__(
            node, 
            graph=graph, 
            name=name, 
            options=options, 
            indent=indent,
            nodenamefunc=nodenamefunc, 
            nodeattrfunc=nodeattrfunc,
            edgeattrfunc=edgeattrfunc, 
            edgetypefunc=edgetypefunc
        )
        self.greedy_nodes = greedy_nodes
        self.probabilities = probabilities
        self.prediction_color = prediction_color
        self.non_prediction_color = non_prediction_color
        self.threshold = threshold
        self.excluded_nodes = set()

    def _default_nodeattrfunc(self, node):
        return f"color={self.prediction_color}" if node in self.greedy_nodes else ""

    def _default_edgeattrfunc(
        self,
        parent, 
        child,
    ):
        color = self.prediction_color if child in self.greedy_nodes else self.non_prediction_color
        label = f"{self.probabilities[child.index_in_softmax_layer]:.2f}" if child.index_in_softmax_layer is not None else "x"
        return f"label={label},color={color}"

    def exclude_node(self, node) -> bool:
        if node in self.excluded_nodes:
            return True
        
        if node.index_in_softmax_layer is None:
            exclude_node = node.parent in self.excluded_nodes
        else:
            include_node = node.is_root or node in self.greedy_nodes or self.probabilities[node.index_in_softmax_layer] >= self.threshold
            exclude_node = not include_node
            
        if exclude_node:
            self.excluded_nodes.add(node)
        return exclude_node

    def _DotExporter__iter_nodes(self, indent, nodenamefunc, nodeattrfunc, *args, **kwargs):
        for node in PreOrderIter(self.node, maxlevel=self.maxlevel):
            if self.exclude_node(node):
                continue
            nodename = nodenamefunc(node)
            nodeattr = nodeattrfunc(node)
            nodeattr = " [%s]" % nodeattr if nodeattr is not None else ""
            yield '%s"%s"%s;' % (indent, DotExporter.esc(nodename), nodeattr)

    def _DotExporter__iter_edges(self, indent, nodenamefunc, edgeattrfunc, edgetypefunc, *args, **kwargs):
        maxlevel = self.maxlevel - 1 if self.maxlevel else None
        for node in PreOrderIter(self.node, maxlevel=maxlevel):
            nodename = nodenamefunc(node)
            for child in node.children:
                if self.exclude_node(child):
                    continue

                childname = nodenamefunc(child)
                edgeattr = edgeattrfunc(node, child)
                edgetype = edgetypefunc(node, child)
                edgeattr = " [%s]" % edgeattr if edgeattr is not None else ""
                yield '%s"%s" %s "%s"%s;' % (indent, DotExporter.esc(nodename), edgetype,
                                             DotExporter.esc(childname), edgeattr)
