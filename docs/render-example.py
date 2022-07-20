from pathlib import Path
from hierarchicalsoftmax import SoftmaxNode

root = SoftmaxNode("root")
a = SoftmaxNode("a", parent=root)
aa = SoftmaxNode("aa", parent=a)
ab = SoftmaxNode("ab", parent=a)
b = SoftmaxNode("b", parent=root)
ba = SoftmaxNode("ba", parent=b)
bb = SoftmaxNode("bb", parent=b)

root.render(print=True)
root.render(filepath=Path(__file__).parent/"images/example-tree.svg")
