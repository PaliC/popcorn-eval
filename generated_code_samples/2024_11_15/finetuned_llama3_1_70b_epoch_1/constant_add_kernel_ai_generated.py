import triton
import triton.language as tl
import torch
import inspect
from _helper_functions import _test_puzzle
from jaxtyping import Float32, Int32
from torch import Tensor

# taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles

def constant_add(x: Float32[Tensor, "32"]) -> Float32[Tensor, "32"]:
    return x + 10

@triton.jit
def constant_add_kernel(x_ptr, z_ptr, N0:
    x_ptr = tl.program_id(0) + N0
    z_ptr = tl.program_id(0) + N0 +  N0
    y = tl.load(x_ptr)
    y +=  tl.load(z_ptr)
    tl.store(z_ptr, y)
    return

```



```python
@triton.jit
def constant_add_kernel(x_ptr, z_ptr):
    x_ptr = tl.program_id(0) + N0
    z_ptr = tl.program_id(0) + N0 + N0
    y = tl.load(x_ptr)
    y += tl.load(z_ptr)
    tl.store(z_ptr, y)
```



```python
@triton.jit
def constant_add_kernel(x_ptr, z_ptr):
    x_ptr = tl.program_id(0) + N0
    z_ptr = tl.program_id(0) + N0 + N0
    y = tl.load(x_ptr)
    y += tl.load(z_ptr)
    tl.store(z_ptr, y)
```



```python
@triton.jit
def constant_add_kernel(x_ptr, z_ptr):
    x_ptr = tl.program_id(0) + N0
    z_ptr = tl.program_id(0) + N0 + N0
    y = tl.load(x_ptr)
    y += tl.load(z_ptr)
    tl.store(z_ptr)
```



```python
@triton.jit
def constant_add_kernel(x_ptr, z_ptr):
    x_ptr = tl.program_id(0) + N0
    z_ptr = tl.program_id(0) + N0 + N0
    y = tl.load(x_ptr)
    y += tl.load(z_ptr)
    tl.store(z_ptr)
```



```python
@triton.jit
def constant_add_kernel(x_ptr z_ptr):
    x_ptr = tl.program_id(0) + N0
    z_ptr = tl.program_id(0) + N0 + N0
    y = tl.load(x_ptr)
    y += tl.load(z_ptr)
    tl.store(z_ptr)
```



```python
@triton.jit
def constant_add_kernel(x_ptr z_ptr):
    x_ptr = tl.program_id(0) + N0
    z_ptr = tl.program_id(0) + N0 + N0
    y = tl.load(x_ptr)
    y += tl.load(z_ptr)
    tl.store(z_ptr)
```



```python
@triton.jit
def constant_add_kernel(x_ptr z_ptr):
    x_ptr = tl.program_id(0) + N0
    z_ptr = tl.program_id(0) + N0 + N0
    y = tl.load(x_ptr)
    y += tl.load(z_ptr)
    tl.store(z_ptr)
```



```python
@triton.jit
def constant_add_kernel(x_ptr z_ptr):
    x_ptr = tl.program_id(0) + N0
    z_ptr = tl.program_id(0) + N0 + N0
    y = tl.load(x_ptr)
    y += tl.load(z_ptr)
    tl.store(z_ptr)
```



```python
@triton.jit
def constant_add_kernel(x_ptr z_ptr):
    x_ptr = tl.program_id(0) + N0
    z_ptr = tl.program_id(0) + N0 + N0
    y = tl.load(x_ptr)
    y += tl.load(z_ptr)
    tl.store(z_ptr)
```



```python
@triton.jit
def constant_add_kernel(x_ptr z_ptr):
    x_ptr = tl.program_id(0) + N0
    z_ptr = tl.program_id(0) + N0 + N0
    y = tl.load(x_ptr)
    y += tl.load(z_ptr)
    tl.store(z_ptr)
```



```python
@triton.jit
def constant_add_kernel(x_ptr z_ptr):
    x_ptr = tl.program_id(0) + N0
    z_ptr = tl.program_id(0) + N0 + N0
    y = tl.load(x_ptr)
    y += tl.load(z_ptr)
    tl.store(z_ptr)
```



```python
@triton.jit
def constant_add_kernel(x_ptr z_ptr):
    x_ptr = tl.program_id(0) + N0
    z_ptr = tl.program_id(0) + N0 + N0
    y = tl.load(x_ptr)
    y += tl.load(z_ptr)
    tl.store(z_ptr)
```




    
if __name__ == "__main__":
    _test_puzzle(constant_add_kernel, constant_add, nelem={"N0": 32})
