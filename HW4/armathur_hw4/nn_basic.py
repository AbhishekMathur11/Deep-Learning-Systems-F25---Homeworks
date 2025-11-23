"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).reshape((1, out_features)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        # ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        y = ops.matmul(X, self.weight)
        if self.bias is not None:
            y = y + ops.broadcast_to(self.bias, y.shape)
        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        # ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return ops.reshape(X, (X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)  
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        # ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        y_one_hot = init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype)
        logits_for_answers = ops.summation(logits * y_one_hot, axes=(1,))
        log_sum_exp = ops.logsumexp(logits, axes=(1,))
        loss_per_sample = -logits_for_answers + log_sum_exp
        
        return ops.summation(loss_per_sample) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        # ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if self.training:
            batch_mean = ops.summation(x, axes=(0,)) / x.shape[0]
            x_mid = x - ops.broadcast_to(ops.reshape(batch_mean, (1, self.dim)), x.shape)
            batch_var = ops.summation(x_mid * x_mid, axes=(0,)) / x.shape[0]

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.detach()
 
            mean_final = batch_mean
            var_final = batch_var
        else:
            mean_final = self.running_mean
            var_final = self.running_var
            x_mid = x - ops.broadcast_to(ops.reshape(mean_final, (1, self.dim)), x.shape)
        
        std = ops.power_scalar(var_final + self.eps, 0.5)
        x_norm = x_mid / ops.broadcast_to(ops.reshape(std, (1, self.dim)), x.shape)

        weight_mod = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
        bias_mod = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        return x_norm * weight_mod + bias_mod

        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        # ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        mean = ops.summation(x, axes=(1,)) / self.dim
        mean = ops.reshape(mean, (x.shape[0], 1))
        x_mid = x - ops.broadcast_to(mean, x.shape)
        var = ops.summation(x_mid*x_mid, axes=(1,)) / self.dim
        var = ops.reshape(var, (x.shape[0], 1))

        std = ops.power_scalar(var + self.eps, 0.5)
        x_norm = x_mid / ops.broadcast_to(std, x.shape)

        weight_mod = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
        bias_mod = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        return x_norm * weight_mod + bias_mod
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        # ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if self.training:
            mask = init.randb(*x.shape, p=1 - self.p, device=x.device, dtype=x.dtype)
            return (x * mask) / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        # ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return self.fn(x) + x
        ### END YOUR SOLUTION




class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))