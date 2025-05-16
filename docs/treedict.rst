TreeDict
========

The ``TreeDict`` class is a convenience wrapper for associating keys with nodes in a hierarchical tree. 
Each key is mapped to a node in the tree and assigned to a specific partition. This is useful in tasks 
such as machine learning classification where validation data is grouped by partition, or in hierarchical 
softmax applications.

TreeDict extends the regular Python dictionary class and provides additional methods to:

- Add and retrieve nodes associated with keys
- Track partition membership
- Render or visualize the tree structure
- Serialize and deserialize the tree and key mapping
- Truncate the tree to a specific depth
- Output tree summaries in human-readable and visual form

TreeDict objects use a classification tree based on ``SoftmaxNode``.

.. code-block:: python

   from hierarchicalsoftmax import TreeDict, SoftmaxNode

    my_tree = SoftmaxNode("root")
    a = SoftmaxNode("a", parent=my_tree)
    aa = SoftmaxNode("aa", parent=a)
    ab = SoftmaxNode("ab", parent=a)
    b = SoftmaxNode("b", parent=my_tree)
    ba = SoftmaxNode("ba", parent=b)
    bb = SoftmaxNode("bb", parent=b)

    tree = TreeDict(my_tree)

Add keys to the TreeDict using the ``add`` method: 

.. code-block:: python

   from hierarchicalsoftmax import TreeDict

   tree = TreeDict()
   tree.add("item_aa_1", aa, partition=0)
   tree.add("item_aa_2", aa, partition=0)

Now you can retrieve the node ID and the partition associated with a key:

.. code-block:: python

   node_detail = tree["item_aa_1"]
   print(node_detail.partition)
   print(node_detail.node_id)

You can also get the actual node object:

.. code-block:: python

   node = tree.node("item_aa_1")


Classes
-------

.. autoclass:: hierarchicalsoftmax.treedict.TreeDict
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: hierarchicalsoftmax.treedict.NodeDetail
   :members:
   :undoc-members:

Command Line Interface
----------------------

The CLI is provided through the Typer app and installed as the command ``treedict``.

.. code-block:: console

   $ treedict --help

This CLI provides several subcommands:

``keys``
   Print the list of keys in a TreeDict. Optionally filter by partition.

   .. code-block:: console

      $ treedict keys data/tree.pkl
      $ treedict keys data/tree.pkl --partition 0

``render``
   Render the tree structure to the console or to a file. You may include node counts or per-partition counts.

   .. code-block:: console

      $ treedict render data/tree.pkl --count
      $ treedict render data/tree.pkl --partition-counts --output out.txt

``count``
   Print the total number of keys in the TreeDict.

   .. code-block:: console

      $ treedict count data/tree.pkl

``sunburst``
   Generate a sunburst plot of the tree using Plotly. You may save to a file or display it interactively.

   .. code-block:: console

      $ treedict sunburst data/tree.pkl --output tree.html
      $ treedict sunburst data/tree.pkl --show

``truncate``
   Truncate the tree to a specified maximum depth and save the new TreeDict to a file.

   .. code-block:: console

      $ treedict truncate data/tree.pkl 3 out/tree-truncated.pkl

``layer-size``
   Print the size of the neural network output layer required for classifying against the current tree.

   .. code-block:: console

      $ treedict layer-size data/tree.pkl

``pickle-tree``
   Serialize only the classification tree (excluding keys) to a pickle file.

   .. code-block:: console

      $ treedict pickle-tree data/tree.pkl out/tree-only.pkl
