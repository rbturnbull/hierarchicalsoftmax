from hierarchicalsoftmax.tensors import LazyLinearTensor
import torch
import unittest


class TestLazyLinearTensor(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.in_features = 5
        self.out_features = 11
        self.weight = torch.nn.Parameter(torch.arange(self.out_features * self.in_features).reshape(self.out_features, self.in_features).float())
        self.bias = torch.nn.Parameter(torch.arange(self.out_features).float())
        self.input = torch.arange(self.batch_size * self.in_features).reshape(self.batch_size, self.in_features).float()
        self.tensor = LazyLinearTensor(self.input, self.weight, self.bias)

    def test_add(self):
        result = self.tensor + 1
        expected = torch.matmul(self.input, self.weight.t()) + self.bias + 1
        assert torch.allclose(result, expected)

    def test_mul(self):
        result = self.tensor * 2
        expected = torch.matmul(self.input, self.weight.t()) * 2 + self.bias * 2
        assert torch.allclose(result, expected)

    def test_sub(self):
        result = self.tensor - 1
        expected = torch.matmul(self.input, self.weight.t()) + self.bias - 1
        assert torch.allclose(result, expected)

    def test_shape(self):
        assert self.tensor.shape == (self.batch_size, self.out_features)

    def test_get_item(self):
        result = self.tensor[0]
        assert isinstance(result, LazyLinearTensor)
        assert result.shape == (self.out_features,)

    def test_slice(self):
        result = self.tensor[:,:2]
        assert not isinstance(result, LazyLinearTensor)

    def test_get_item_slice(self):
        result = self.tensor[0]
        assert torch.allclose(result[:7], self.tensor.result[0,:7])
        assert torch.allclose(result[7:], self.tensor.result[0,7:])

    def test_get_item(self):
        assert len(self.tensor) == self.batch_size
        assert len(self.tensor[0]) == self.out_features

    def test_iter_slice(self):
        for i, tensor in enumerate(self.tensor):
            assert torch.allclose(tensor[:2], self.tensor[i][:2])

    def test_str(self):
        assert str(self.tensor) == "LazyLinearTensor (shape=(2, 11))"
    
    def test_repr(self):
        assert repr(self.tensor) == "LazyLinearTensor (shape=(2, 11))"

    def test_truediv(self):
        result = self.tensor / 2
        expected = torch.matmul(self.input, self.weight.t()) / 2 + self.bias / 2
        assert torch.allclose(result, expected)

    def test_matmul(self):
        result = self.tensor @ torch.arange(self.out_features).float()
        expected = torch.matmul(self.tensor.result, torch.arange(self.out_features).float())
        assert torch.allclose(result, expected)

    def test_radd(self):
        result = 1 + self.tensor
        expected = torch.matmul(self.input, self.weight.t()) + self.bias + 1
        assert torch.allclose(result, expected)

    def test_rsub(self):
        result = 1 - self.tensor
        expected = 1 - (torch.matmul(self.input, self.weight.t()) + self.bias)
        assert torch.allclose(result, expected)    

    def test_rmul(self):
        result = 2 * self.tensor
        expected = 2 * (torch.matmul(self.input, self.weight.t()) + self.bias)
        assert torch.allclose(result, expected)    

    def test_rtruediv(self):
        result = 2 / self.tensor
        expected = 2 / (torch.matmul(self.input, self.weight.t()) + self.bias)
        assert torch.allclose(result, expected)    
    
    def test_rmatmul(self):
        size = 9
        matrix = torch.arange(self.batch_size*size).reshape(size, self.batch_size).float()
        result = matrix @ self.tensor
        expected = matrix @ (torch.matmul(self.input, self.weight.t()) + self.bias)
        assert torch.allclose(result, expected)

    def test_index_error(self):
        with self.assertRaises(IndexError):
            self.tensor[0,0,5]
