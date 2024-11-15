import triton
import triton.language as tl
import torch
import inspect
from _helper_functions import _test_puzzle
import jaxtyping
from jaxtyping import Float32, Int32
from torch import Tensor

# taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles

def outer_vector_add(x: Float32[Tensor, "32"], y: Float32[Tensor, "32"]) -> Float32[Tensor, "32 32"]:
    return x[None, :] + y[:, None]

@triton.jit
def add_kernel(x_ptr, y_ptr):
    x_ptr = tl.load(x_ptr + tl.arange(0, tl.load(y_ptr + tl.arange(0)
    return x_ptr + y_ptr
```

```python
@triton.jit
def tl_add_kernel(x_ptr, y_ptr):
    return tl.load(x_ptr + tl.load(y_ptr))
```

```python
@triton.jit
def tl_vector_add_kernel(x_ptr0, y_ptr0):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0)
    y_ptr0 = tl.load(y_ptr0 + tl.arange(0))
    return x_ptr0 + y_ptr0
```

```python
@triton.jit
def tl_vector_add_kernel(x_ptr0: tl.constexpr):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0))
    return x_ptr0
```

```python
@triton.jit
def vector_add_kernel(x_ptr0: tl.constexpr):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0))
    return x_ptr0
```

```python
@triton.jit
def tl_vector_add_kernel(x_ptr0: tl.constexpr):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0))
    return x_ptr0
```

```python
@triton.jit
def tl_vector_add_kernel(x_ptr0: tl.constexpr):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0))
    return x_ptr0
```

```python
@triton.jit
def vector_add_kernel(x_ptr0: tl.constexpr):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0))
    return x_ptr0
```

```python
@triton.jit
def tl_vector_add_kernel(x_ptr0: tl.constexpr):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0))
    return x_ptr0
```

```python
@triton.jit
def vector_add_kernel(x_ptr0: tl.constexpr):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0))
    return x_ptr0
```

```python
@triton.jit
def tl_vector_add_kernel(x_ptr0: tl.constexpr):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0))
    return x_ptr0
```

```python
@triton.jit
def vector_add_kernel(x_ptr0: tl.constexpr):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0))
    return x_ptr0
```

```python
@triton.jit
def tl_vector_add_kernel(x_ptr0: tl.constexpr):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0))
    return x_ptr0
```

```python
@triton.jit
def vector_add_kernel(x_ptr0: tl.constexpr):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0))
    return x_ptr0
```

```python
@triton.jit
def tl_vector_add_kernel(x_ptr0: tl.constexpr):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0))
    return x_ptr0
```

```python
@triton.jit
def vector_add_kernel(x_ptr0: tl.constexpr):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0))
    return x_ptr0
```

```python
@triton.jit
def tl_vector_add_kernel(x_ptr0: tl.constexpr):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0))
    return x_ptr0
```

```python
@triton.jit
def vector_add_kernel(x_ptr0: tl.constexpr):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0))
    return x_ptr0
```

```python
@triton.jit
def tl_vector_add_kernel(x_ptr0: tl.constexpr):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0))
    return x_ptr0
```

```python
@triton.jit
def vector_add_kernel(x_ptr0: tl.constexpr):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0))
    return x_ptr0
```

```python
@triton.jit
def tl_vector_add_kernel(x_ptr0: tl.constexpr):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0))
    return x_ptr0
```

```python
@triton.jit
def vector_add_kernel(x_ptr0: tl.constexpr):
    x_ptr0 = tl.load(x_ptr0 + tl.arange(0))
    return x_ptr0
```


    
if __name__ == "__main__":
    _test_puzzle(outer_vector_add_kernel, outer_vector_add, nelem={"N0": 32, "N1": 32})
