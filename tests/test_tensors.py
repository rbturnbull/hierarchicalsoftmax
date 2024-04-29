from hierarchicalsoftmax.tensors import HierarchicalSoftmaxTensor
import torch
import unittest


class TestHierarchicalSoftmaxTensor(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.in_features = 3
        self.out_features = 5
        self.weight = torch.nn.Parameter(torch.arange(self.out_features * self.in_features).reshape(self.out_features, self.in_features).float())
        self.bias = torch.nn.Parameter(torch.arange(self.out_features).float())
        self.input = torch.arange(self.batch_size * self.in_features).reshape(self.batch_size, self.in_features).float()
        self.tensor = HierarchicalSoftmaxTensor(self.input, self.weight, self.bias)

    def test_add(self):
        result = self.tensor + 1
        expected = torch.matmul(self.input, self.weight.t()) + self.bias + 1
        self.assertTrue(torch.allclose(result, expected))

    def test_sub(self):
        result = self.tensor - 1
        expected = torch.matmul(self.input, self.weight.t()) + self.bias - 1
        self.assertTrue(torch.allclose(result, expected))

    def test_shape(self):
        assert self.tensor.shape == (self.batch_size, self.out_features)

    def test_get_item(self):
        result = self.tensor[0]
        assert isinstance(result, HierarchicalSoftmaxTensor)
        assert result.shape == (self.out_features,)

        