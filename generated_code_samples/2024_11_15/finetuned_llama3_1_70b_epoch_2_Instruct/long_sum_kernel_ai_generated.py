import triton
import triton.language as tl
import torch
import inspect
from _helper_functions import _test_puzzle
from jaxtyping import Float32, Int32
from torch import Tensor

# taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles

def long_sum_spec(x: Float32[Tensor, "4 200"]) -> Float32[Tensor, "4"]:
    return x.sum(1)

@triton.jit
def _long_sum_kernel(x_ptr, z_ptr, N0: tl.constexpr
 T: tl.constexpr
 B0: tl.constexpr
 B1: tl.constexpr):
    x_ptr = x_ptr + B0 * T
    z_ptr = z_ptr + B1 * T
    tl.store(x_ptr)
    tl.store(z_ptr)

```python
@triton.jit
def _long_sum_kernel(x_ptr, z_ptr N0: tl.constexpr
 T: tl.constexpr
 B0: tl.constexpr
 B1: tl.constexpr):
    x_ptr = x_ptr + B0 * T
    z_ptr = z_ptr + B1 * T
    tl.store(x_ptr)
    tl.store(z_ptr)

```python
@triton.jit
def _long_sum_kernel(x_ptr z_ptr N0: tl.constexpr
 T: tl.constexpr
 B0: tl.constexpr
 B1: tl.constexpr):
    x_ptr = x_ptr + B0 * T
    z_ptr = z_ptr + B1 * T
    tl.store(x_ptr)
    tl.store(z_ptr)

```python
@triton.jit
def _long_sum_kernel(x_ptr z_ptr N0: tl.constexpr
 T: tl.constexpr
 B0: tl.constexpr
 B1: tl.constexpr):
    x_ptr = x_ptr + B0 * T
    z_ptr = z_ptr + B1 * T
    tl.store(x_ptr)
    tl.store(z_ptr)

```python
@triton.jit
def _long_sum_kernel(x_ptr z_ptr N0: tl.constexpr
 T: tl.constexpr
 B0: tl.constexpr
 B1: tl.constexpr):
    x_ptr = x_ptr + B0 * T
    z_ptr = z_ptr + B1 * T
    tl.store(x_ptr)
    tl.store(z_ptr)

```python
@triton.jit
def _long_sum_kernel(x_ptr z_ptr N0: tl.constexpr
 T: tl.constexpr
 B0: tl.constexpr
 B1: tl.constexpr):
    x_ptr = x_ptr + B0 * T
    z_ptr = z_ptr + B1 * T
    tl.store(x_ptr)
    tl.store(z_ptr)

```python
@triton.jit
def _long_sum_kernel(x_ptr z_ptr N0: tl.constexpr
 T: tl.constexpr
 B0: tl.constexpr
 B1: tl.constexpr):
    x_ptr = x_ptr + B0 * T
    z_ptr = z_ptr + B1 * T
    tl.store(x_ptr)
    tl.store(z_ptr)

```python
@triton.jit
def _long_sum_kernel(x_ptr z_ptr N0: tl.constexpr
 T: tl.constexpr
 B0: tl.constexpr
 B1: tl.constexpr):
    x_ptr = x_ptr + B0 * T
    z_ptr = z_ptr + B1 * T
    tl.store(x_ptr)
    tl.store(z_ptr)

```python
@triton.jit
def _long_sum_kernel(x_ptr z_ptr N0: tl.constexpr
 T: tl.constexpr
 B0: tl.constexpr
 B1: tl.constexpr):
    x_ptr = x_ptr + B0 * T
    z_ptr = z_ptr + B1 * T
    tl.store(x_ptr)
    tl.store(z_ptr)

```python
@triton.jit
def _long_sum_kernel(x_ptr z_ptr N0: tl.constexpr
 T: tl.constexpr
 B0: tl.constexpr
 B1: tl.constexpr):
    x_ptr = x_ptr + B0 * T
    z_ptr = z_ptr + B1 * T
    tl.store(x_ptr)
    tl.store(z_ptr)

```python
@triton.jit
def _long_sum_kernel(x_ptr z_ptr N0: tl.constexpr
 T: tl.constexpr
 B0: tl.constexpr
 B1: tl.constexpr):
    x_ptr = x_ptr + B0 * T
    z_ptr = z_ptr + B1 * T
    tl.store(x_ptr)
    tl.store(z_ptr)

```python
@triton.jit
def _long_sum_kernel(x_ptr z_ptr N0: tl.constexpr
 T: tl.constexpr
 B0: tl.constexpr
 B1: tl.constexpr):
    x_ptr = x_ptr + B0 * T
    z_ptr = z_ptr + B1 * T
    tl.store(x_ptr)
    tl.store(z_ptr)


    
if __name__ == "__main__":
    _test_puzzle(long_sum_kernel, long_sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})
