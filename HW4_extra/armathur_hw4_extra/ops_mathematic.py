"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        base, exponent = node.inputs
        grad_base = out_grad * exponent * power(base, exponent - 1)
        grad_exponent = out_grad * power(base, exponent) * log(base)
        return grad_base, grad_exponent
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * self.scalar * power_scalar(node.inputs[0], self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / (rhs ** 2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            ndim = len(a.shape)
            axes = list(range(ndim))
            axes[-2], axes[-1] = axes[-1], axes[-2]
            return a.permute(tuple(axes))
        else:
            ndim = len(a.shape)
            axes = list(range(ndim))
            axes[self.axes[0]], axes[self.axes[1]] = axes[self.axes[1]], axes[self.axes[0]]
            return a.permute(tuple(axes))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if not a.is_compact():
            a = a.compact()
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.broadcast_to(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        output_shape = self.shape
        
        axes_to_sum = []
        
        ndim_diff = len(output_shape) - len(input_shape)
        if ndim_diff > 0:
            axes_to_sum.extend(range(ndim_diff))
        
        for i in range(len(input_shape)):
            output_axis = i + ndim_diff
            if input_shape[i] == 1 and output_shape[output_axis] > 1:
                axes_to_sum.append(output_axis)
        
        grad = out_grad
        if axes_to_sum:
            for axis in sorted(axes_to_sum, reverse=True):
                grad = summation(grad, axes=(axis,))
        
        return reshape(grad, input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.sum(axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        
        if self.axes is None:
            new_shape = [1] * len(input_shape)
        else:
            new_shape = list(input_shape)
            axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            for axis in axes:
                new_shape[axis] = 1
        
        grad_reshaped = reshape(out_grad, tuple(new_shape))
        
        return broadcast_to(grad_reshaped, input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lgrad = matmul(out_grad, transpose(rhs))
        rgrad = matmul(transpose(lhs), out_grad)
        
        if len(lhs.shape) < len(lgrad.shape):
            lgrad = summation(lgrad, axes=tuple(range(len(lgrad.shape) - len(lhs.shape))))
        if len(rhs.shape) < len(rgrad.shape):
            rgrad = summation(rgrad, axes=tuple(range(len(rgrad.shape) - len(rhs.shape))))
        
        return lgrad, rgrad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.log()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.exp()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.maximum(0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        mask = Tensor(node.inputs[0].realize_cached_data() > 0, device=out_grad.device, dtype=out_grad.dtype)
        return out_grad * mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        tanh_squared = tanh(node.inputs[0]) ** 2
        return out_grad * ((-tanh_squared) + 1)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        arrays = [arr for arr in args]
        shape = arrays[0].shape
        n = len(arrays)
        
        new_shape = list(shape)
        new_shape.insert(self.axis, n)
        
        device = arrays[0].device
        out = device.empty(tuple(new_shape), dtype=arrays[0].dtype)
        
        for i, arr in enumerate(arrays):
            slices = [slice(None)] * len(new_shape)
            slices[self.axis] = i
            out[tuple(slices)] = arr
        
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        n = A.shape[self.axis]
        result = []
        
        for i in range(n):
            slices = [slice(None)] * len(A.shape)
            slices[self.axis] = i
            arr = A[tuple(slices)]
            
            arr = arr.compact()
            
            new_shape = list(A.shape)
            new_shape.pop(self.axis)
            arr = arr.reshape(tuple(new_shape))
            
            result.append(arr)
        
        return tuple(result)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0:
            return a
        
        new_shape = list(a.shape)
        for axis in self.axes:
            if axis < len(a.shape):
                new_shape[axis] = a.shape[axis] * (self.dilation + 1)
        
        from ..backend_ndarray.ndarray import NDArray
        out = NDArray.make(tuple(new_shape), device=a.device)
        out.fill(0)
        
        slices = []
        for i in range(len(a.shape)):
            if i in self.axes and i < len(a.shape):
                slices.append(slice(0, new_shape[i], self.dilation + 1))
            else:
                slices.append(slice(0, new_shape[i]))
        
        out[tuple(slices)] = a
        
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, axes=self.axes, dilation=self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0:
            return a
        
        slices = []
        for i in range(len(a.shape)):
            if i in self.axes and i < len(a.shape):
                slices.append(slice(0, a.shape[i], self.dilation + 1))
            else:
                slices.append(slice(0, a.shape[i]))
        
        return a[tuple(slices)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, axes=self.axes, dilation=self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, Z, weight):
        ### BEGIN YOUR SOLUTION
        Z_padded = Z.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        
        N, H, W, C_in = Z_padded.shape
        K, _, _, C_out = weight.shape
        Ns, Hs, Ws, Cs = Z_padded.strides
        
        H_out = (H - K) // self.stride + 1
        W_out = (W - K) // self.stride + 1
        
        inner_dim = K * K * C_in
        
        from ..backend_ndarray.ndarray import NDArray
        im_col = Z_padded.as_strided(
            shape=(N, H_out, W_out, K, K, C_in),
            strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)
        ).compact()
        
        im_col = im_col.reshape((N * H_out * W_out, inner_dim))
        
        weight = weight.compact().reshape((inner_dim, C_out))
        
        out = im_col @ weight
        
        return out.reshape((N, H_out, W_out, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        K, _, C_in, C_out = W.shape
        
        if self.stride > 1:
            out_grad_dilated = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1)
        else:
            out_grad_dilated = out_grad
        
        W_flipped = flip(flip(W, axes=(0,)), axes=(1,))
        W_flipped = transpose(W_flipped, axes=(2, 3))
        
        padding_X = K - 1 - self.padding
        
        X_grad = conv(out_grad_dilated, W_flipped, stride=1, padding=padding_X)
        
        X_permuted = transpose(X, axes=(0, 3))
        
        out_grad_permuted = transpose(out_grad_dilated, axes=(0, 1))
        out_grad_permuted = transpose(out_grad_permuted, axes=(1, 2))
        
        W_grad = conv(X_permuted, out_grad_permuted, stride=1, padding=self.padding)
        
        W_grad = transpose(W_grad, axes=(0, 1))
        W_grad = transpose(W_grad, axes=(1, 2))
        
        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


