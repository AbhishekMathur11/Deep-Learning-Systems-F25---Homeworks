from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(axis=-1, keepdims=True)
        exp_z_shifted = (Z - max_z).exp()
        sum_exp = exp_z_shifted.sum(axis=-1, keepdims=True)
        log_sum_exp = sum_exp.log()
        return Z - max_z - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        softmax = exp(node)
        sum_grad = summation(out_grad, axes=-1)
        sum_grad = reshape(sum_grad, shape=sum_grad.shape + (1,))
        return out_grad - softmax * sum_grad
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(axis=self.axes, keepdims=True)
        exp_z_shifted = (Z - max_z.broadcast_to(Z.shape)).exp()
        sum_exp = exp_z_shifted.sum(axis=self.axes, keepdims=True)
        log_sum_exp = sum_exp.log()
        result = max_z + log_sum_exp
        
        if self.axes is not None:
            axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            new_shape = [Z.shape[i] for i in range(len(Z.shape)) if i not in axes]
            result = result.reshape(tuple(new_shape))
        else:
            result = result.reshape((1,))
            result = result.reshape(())
        
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        output_shape = node.shape
        
        out_grad_data = out_grad.realize_cached_data().compact()
        from ..autograd import Tensor
        out_grad = Tensor.make_const(out_grad_data)
        
        if self.axes is not None:
            new_shape = list(input_shape)
            if isinstance(self.axes, int):
                axes = (self.axes,)
            else:
                axes = self.axes
            for axis in axes:
                new_shape[axis] = 1
            out_grad_reshaped = reshape(out_grad, new_shape)
            
            node_data = node.realize_cached_data().compact()
            node_compact = Tensor.make_const(node_data)
            node_reshaped = reshape(node_compact, new_shape)
        else:
            new_shape = [1] * len(input_shape)
            out_grad_reshaped = reshape(out_grad, new_shape)
            
            node_data = node.realize_cached_data().compact()
            node_compact = Tensor.make_const(node_data)
            node_reshaped = reshape(node_compact, new_shape)

        out_grad_broadcasted = broadcast_to(out_grad_reshaped, input_shape)
        node_reshaped_broadcasted = broadcast_to(node_reshaped, input_shape)
        return out_grad_broadcasted * exp(node.inputs[0] - node_reshaped_broadcasted)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)