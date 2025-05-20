================================================================
hierarchicalsoftmax
================================================================

.. start-badges

|pypi badge| |testing badge| |coverage badge| |docs badge| |black badge|

.. |pypi badge| image:: https://img.shields.io/pypi/v/hierarchicalsoftmax?color=blue
    :target: https://pypi.org/project/hierarchicalsoftmax/

.. |testing badge| image:: https://github.com/rbturnbull/hierarchicalsoftmax/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/rbturnbull/hierarchicalsoftmax/actions

.. |docs badge| image:: https://github.com/rbturnbull/hierarchicalsoftmax/actions/workflows/docs.yml/badge.svg
    :target: https://rbturnbull.github.io/hierarchicalsoftmax
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. |coverage badge| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rbturnbull/f99aea7ea203d16edd063a8dd5ed395f/raw/coverage-badge.json
    :target: https://rbturnbull.github.io/hierarchicalsoftmax/coverage/
    
.. end-badges

A Hierarchical Softmax Framework for PyTorch.

Documentation available here: `https://rbturnbull.github.io/hierarchicalsoftmax/ <https://rbturnbull.github.io/hierarchicalsoftmax/>`_.

.. start-quickstart


Installation
==================================

hierarchicalsoftmax can be installed from PyPI:

.. code-block:: bash

    pip install hierarchicalsoftmax


Alternatively, hierarchicalsoftmax can be installed using pip from the git repository:

.. code-block:: bash

    pip install git+https://github.com/rbturnbull/hierarchicalsoftmax.git


Usage
==================================

Build up a hierarchy tree for your categories using the `SoftmaxNode` instances:

.. code-block:: python

    from hierarchicalsoftmax import SoftmaxNode

    root = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

The `SoftmaxNode` class inherits from the `anytree <https://anytree.readthedocs.io/en/latest/index.html>`_ `Node` class 
which means that you can use methods from that library to build and interact with your hierarchy tree.

The tree can be rendered as a string with the `render` method:

.. code-block:: python

    root.render(print=True)

This results in a text representation of the tree::

    root
    ├── a
    │   ├── aa
    │   └── ab
    └── b
        ├── ba
        └── bb

The tree can also be rendered to a file using `graphviz` if it is installed:

.. code-block:: python

    root.render(filepath="tree.svg")

.. image:: https://raw.githubusercontent.com/rbturnbull/hierarchicalsoftmax/main/docs/images/example-tree.svg
    :alt: An example tree rendering.


Then you can add a final layer to your network that has the right size of outputs for the softmax layers.
You can do that manually by setting the output number of features to `root.layer_size`. 
Alternatively you can use the `HierarchicalSoftmaxLinear` or `HierarchicalSoftmaxLazyLinear` classes:

.. code-block:: python

    from torch import nn
    from hierarchicalsoftmax import HierarchicalSoftmaxLinear

    model = nn.Sequential(
        nn.Linear(in_features=20, out_features=100),
        nn.ReLU(),
        HierarchicalSoftmaxLinear(in_features=100, root=root)
    )

Once you have the hierarchy tree, then you can use the `HierarchicalSoftmaxLoss` module:

.. code-block:: python

    from hierarchicalsoftmax import HierarchicalSoftmaxLoss

    loss = HierarchicalSoftmaxLoss(root=root)

Metric functions are provided to show accuracy and the F1 score:

.. code-block:: python

    from hierarchicalsoftmax import greedy_accuracy, greedy_f1_score

    accuracy = greedy_accuracy(predictions, targets, root=root)
    f1 = greedy_f1_score(predictions, targets, root=root)

The nodes predicted from the final layer of the model can be inferred using the `greedy_predictions` function which provides a list of the predicted nodes:

.. code-block:: python

    from hierarchicalsoftmax import greedy_predictions

    outputs = model(inputs)
    inferred_nodes = greedy_predictions(outputs)


Relative contributions to the loss
==================================

The loss for each node can be weighted relative to each other by setting the `alpha` value for each parent node. 
By default the `alpha` value of a node is 1.

For example, the loss for the first level of classification (under the `root` node) will contribute twice as much to the loss than under the `a` or `b` nodes.

.. code-block:: python

    from hierarchicalsoftmax import SoftmaxNode

    root = SoftmaxNode("root", alpha=2.0)
    a = SoftmaxNode("a", parent=root)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=root)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)


Example Usage
==================================

In the documentation, we provide `an example of how to use this package with the CIFAR-10 and CIFAR-100 datasets <https://rbturnbull.github.io/hierarchicalsoftmax/cifar.html>`_.:


TreeDict
==================================

We provide a helper class to create a dictionary where items point to the nodes in the tree.
This is useful for creating a dataloader in a machine learning model.
See the `TreeDict documentation <https://rbturnbull.github.io/hierarchicalsoftmax/cifar.html>`_ for more information.

Label Smoothing
==================================

You can add label smoothing to the loss by setting the `label_smoothing` parameter to any of the nodes.

Focal Loss
==================================

You can use the Focal Loss instead of a basic cross-entropy loss for any of the nodes by setting the `gamma` parameter to any of the nodes.


.. end-quickstart


Credits
==================================

* Robert Turnbull <robert.turnbull@unimelb.edu.au>

