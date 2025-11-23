from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        # ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        logsumexp = LogSumExp(axes=(1,)).compute(Z)
        logsumexp_mod = logsumexp.reshape((-1, 1))
        return Z - logsumexp_mod
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        # ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        Z = node.inputs[0]
        # The gradient of LogSoftmax is: out_grad - softmax(Z) * sum(out_grad, axis=1, keepdims=True)
        
        # Compute softmax(Z) = exp(LogSoftmax(Z))
        log_softmax_z = LogSoftmax()(Z)
        softmax_z = exp(log_softmax_z)
        
        # Sum out_grad along axis=1 and reshape for broadcasting
        out_grad_sum = summation(out_grad, axes=(1,))
        out_grad_sum_reshaped = reshape(out_grad_sum, (out_grad_sum.shape[0], 1))
        
        # Gradient: out_grad - softmax(Z) * sum(out_grad, axis=1, keepdims=True)
        return out_grad - softmax_z * out_grad_sum_reshaped
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        # ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if self.axes is None:
            # Sum over all axes
            max_z = array_api.max(Z)
            z_shifted = Z - max_z
            sum_exp = array_api.sum(array_api.exp(z_shifted))
            return array_api.log(sum_exp) + max_z
        else:
            # Handle specific axes
            max_z = array_api.max(Z, axis=self.axes, keepdims=True)
            z_shifted = Z - max_z
            sum_exp = array_api.sum(array_api.exp(z_shifted), axis=self.axes, keepdims=True)
            result = array_api.log(sum_exp) + max_z
            return array_api.squeeze(result, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        # ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        Z = node.inputs[0]
        
        if self.axes is None:
            # axes=None case: sum over all dimensions
            max_z = array_api.max(Z.realize_cached_data())
            exp_z_minus_max = exp(Z - max_z)
            sum_exp = summation(exp_z_minus_max, axes=None)
            gradient = exp_z_minus_max / sum_exp
            return gradient * out_grad
        else:
            # Specific axes case
            max_z = array_api.max(Z.realize_cached_data(), axis=self.axes, keepdims=True)
            exp_z_minus_max = exp(Z - max_z)
            sum_exp = summation(exp_z_minus_max, axes=self.axes)
            
            shape = list(Z.shape)
            if isinstance(self.axes, int):
                axes = (self.axes,)
            else:
                axes = self.axes
            
            for axis in axes:
                shape[axis] = 1
            
            sum_exp_reshaped = reshape(sum_exp, shape)
            gradient = exp_z_minus_max / sum_exp_reshaped
            out_grad_reshaped = reshape(out_grad, shape)
            return gradient * out_grad_reshaped
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)