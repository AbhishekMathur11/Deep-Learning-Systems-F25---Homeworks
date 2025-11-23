import math
from .init_basic import *
from typing import Any


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    shape = kwargs.pop('shape', (fan_in, fan_out))
    return rand(*shape, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    shape = kwargs.pop('shape', (fan_in, fan_out))
    return randn(*shape, mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # For ReLU, gain = sqrt(2)
    bound = math.sqrt(6.0 / fan_in)
    shape = kwargs.pop('shape', (fan_in, fan_out))
    return rand(*shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    std = math.sqrt(2.0 / fan_in)
    shape = kwargs.pop('shape', (fan_in, fan_out))
    return randn(*shape, mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION
