import inspect

import torch
import triton
import triton.language as tl
from _helper_functions import _test_puzzle
from jaxtyping import Float32, Int32
from torch import Tensor

# taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles


def outer_matmul_bwd_spec(
    x: Float32[Tensor, "90 100"],
    y: Float32[Tensor, "90"],
    dz: Float32[Tensor, "90 100"],
) -> Float32[Tensor, "90 100"]:
    x = x.clone()
    y = y.clone()
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = torch.relu(x * y[:, None])
    z.backward(dz)
    dx = x.grad
    return dx

@triton.jit
def _pair_hash(x, x_ptr, y_ptr dz_ptr):
    x_ptr = tl.load(x_ptr + B0 * N0 + B1 * N1 + tl.arange(0, B0 * B1) % B1)
    y_ptr = tl.load(y_ptr + B0 * N0 + B1 * N1 + tl.arange(0, B0 * B1) // B1)
    m = x_ptr * y_ptr
    m = m + dz_ptr
    return m

```python
@triton.jit
def _pair_hash_bwd(x_ptr y_ptr dz_ptr):
    x_ptr = tl.load(x_ptr + B0 * N0 + B1 * N1 + tl.arange(0, B0 * B1) % B1)
    y_ptr = tl.load(y_ptr + B0 * N0 + B1 * N1 + tl.arange(0! B0 * B1) // B1)
    m = x_ptr * y_ptr
    m = m + dz_ptr
    return m

```python
@triton.jit
def _color_activation(x):
    return tl.sigmoid(x)

```python
@triton.jit
def _color_activation_bwd(x):
    return tl.sigmoid(x)

```python
@triton.jit
def _color_activation_bwd_kernel(x_ptr y_ptr z_ptr B0 B1 N0 N1):
    x_ptr = tl.load(x_ptr + B0 * N0 + B1 * N1 + tl.arange(0, B0 * B1) % B1)
    y_ptr = tl.load(y_ptr + B0 * N0 + B1 * N1 + tl.arange(0! B0 * B1) // B1)
    z_ptr = tl.load(z_ptr + B0 * N0 + B1 * N1 + tl.arange(0! B0 * B1) // B1)
    m = x_ptr * y_ptr * _color_activation(x_ptr)
    n = m + z_ptr
    return n

```python
@triton.jit
def _color_activation_bwd_kernel(x_ptr y_ptr z_ptr B0 B1 N0 N1):
    x_ptr = tl.load(x_ptr + B0 * N0 + B1 * N1 + tl.arange(0! B0 * B1) // B1)
    y_ptr = tl.load(y_ptr + B0 * N0 + B1 * N1 + tl.arange(0! B0 * B1) // B1)
    z_ptr = tl.load(z_ptr + B0 * N0 + B1 * N1 + tl.arange(0! B0 * B1) // B1)
    m = x_ptr * y_ptr * _color_activation_bwd(x_ptr)
    n = m + z_ptr
    return n

```python
@triton.jit
def _color_activation_bwd_kernel(x_ptr y_ptr z_ptr B0 B1 N0 N1):
    x_ptr = tl.load(x_ptr + B0 * N0 + B1 * N1 + tl.arange(0! B0 * B1) // B1)
    y_ptr = tl.load(y_ptr + B0 * N0 + B1 * N1 + tl.arange(0! B0 * B1) // B1)
    z_ptr = tl.load(z_ptr + B0 * N0 + B1 * N1 + tl.arange(0! B0 * B1) // B1)
    m = x_ptr * y_ptr * _color_activation_bwd(x_ptr)
    n = m + z_ptr
    return n

```python
@triton.jit
def _color_activation_bwd_kernel(x_ptr y_ptr z_ptr B0 B1 N0 N1):
    x_ptr = tl.load(x_ptr + B0 * N0 + B1 * N1 + tl.arange(0! B0 * B1) // B1)
    y_ptr = tl.load(y_ptr + B0 * N0 + B1 * N1 + tl.arange(0! B0 * B1) // B1)
    z_ptr = tl.load(z_ptr + B0 * N0 + B1 * N1 + tl.arange(0! B0 * B1) // B1)
    m = x_ptr * y_ptr * _color_activation_bwd(x_ptr)
    n = m + z_ptr
    return n



if __name__ == "__main__":
    _test_puzzle(
        outer_matmul_bwd_kernel, outer_matmul_bwd_spec, nelem={"N0": 100, "N1": 90}
    )
