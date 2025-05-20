from torch import Tensor, Size
from functools import cached_property
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class LazyLinearTensor(Tensor):
    """
    A tensor that is designed to be used with HierarchicalSoftmaxLazyLinear layers.
    """
    @staticmethod
    def __new__(cls, x, weight:Parameter, bias:Parameter, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs)

    def __init__(self, x:Tensor, weight:Parameter, bias:Parameter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input = x
        self.weight = weight
        self.bias = bias

    @cached_property
    def result(self):
        return F.linear(self.input, self.weight, self.bias)

    def __add__(self, other):
        return self.result + other
    
    def __sub__(self, other):
        return self.result - other
    
    def __mul__(self, other):
        return self.result * other
    
    def __truediv__(self, other):
        return self.result / other
    
    def __matmul__(self, other):
        return self.result @ other
    
    def __radd__(self, other):
        return other + self.result
    
    def __rsub__(self, other):
        return other - self.result
    
    def __rmul__(self, other):
        return other * self.result
    
    def __rtruediv__(self, other):
        return other / self.result
    
    def __rmatmul__(self, other):
        return other @ self.result

    def __getitem__(self, index):
        assert isinstance(index, int) or isinstance(index, slice) or isinstance(index, tuple)
        if not isinstance(index, tuple) or isinstance(index, slice):
            index = (index,)

        my_shape = self.shape
        if len(index) < len(my_shape):
            return LazyLinearTensor(self.input[index], weight=self.weight, bias=self.bias)
        if len(index) > len(my_shape):
            raise IndexError(f"Cannot get index '{index}' for LazyLinearTensor of shape {len(my_shape)}")

        input = self.input[index[:-1]]
        weight = self.weight[index[-1]]
        bias = self.bias[index[-1]]      
        return F.linear(input, weight, bias)
    
    @property
    def shape(self) -> Size:
        return Size( self.input.shape[:-1] + (self.weight.shape[0],) )

    def __str__(self) -> str:
        return f"LazyLinearTensor (shape={tuple(self.shape)})"
    
    def __repr__(self) -> str:
        return str(self)
    
    def __len__(self) -> int:
        return self.shape[0]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def float(self):
        x = super().float()
        x.input = self.input.float()
        x.weight = self.weight.float()
        x.bias = self.bias.float()
        return x
    
    def half(self):
        x = super().half()
        x.input = self.input.half()
        x.weight = self.weight.half()
        x.bias = self.bias.half()
        return x    
    
    def to(self, *args, **kwargs):
        x = super().to(*args, **kwargs)
        x.input = self.input.to(*args, **kwargs)
        x.weight = self.weight.to(*args, **kwargs)
        x.bias = self.bias.to(*args, **kwargs)
        return x
