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

def outer_vector_add(x: Float32[Tensor, "32"], y: Float32[Tensor])
    return x[None] + y[:, None]
```

## Writing Kernels
The triton kernel should have the following functionality
```python
def outer_vector_add(x: Float32[Tensor])
    return x[None] + x[:, None]
```

## Writing Kernels
The triton kernel should have the following functionality
```python
def outer_vector_add(x: Float32[Tensor])
    return x[None] + x[None]
```

## Writing Kernels
The triton kernel should have the following functionality
```python
def outer_vector_add(x: Float32[Tensor])
    return x[None] + x[None] + x[None]
```

## Writing Kernels
The triton kernel should have the following functionality
```python
def outer_vector_add(x: Float32[Tensor])
    return x[None] + x[None] + x[None] + x[None]
```

## Writing Kernels
The triton kernel should have the following functionality
```python
def outer_vector_add(x: Float32[Tensor])
    return x[None] + x[None] + x[None] + x[None] + x[None]
```

## Writing Kernels
The triton kernel should have the following functionality
```python
def outer_vector_add(x: Float32[Tensor])
    return x[None] + x[None] + x[None] + x[None] + x[None] + x[None]
```

## Writing Kernels
The triton kernel should have the following functionality
```python
def outer_vector_add(x: Float32[Tensor])
    return x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None]
```

## Writing Kernels
The triton kernel should have the following functionality
```python
def outer_vector_add(x: Float32[Tensor])
    return x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None]
```

## Writing Kernels
The triton kernel should have the following functionality
```python
def outer_vector_add(x: Float32[Tensor])
    return x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None]
```

## Writing Kernels
The triton kernel should have the following functionality
```python
def outer_vector_add(x: Float32[Tensor])
    return x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None]
```

## Writing Kernels
The triton kernel should have the following functionality
```python
def outer_vector_add(x: Float32[Tensor])
    return x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None]
```

## Writing Kernels
The triton kernel should have the following functionality
```python
def outer_vector_add(x: Float32[Tensor])
    return x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None]
```

## Writing Kernels
The triton kernel should have the following functionality
```python
def outer_vector_add(x: Float32[Tensor])
    return x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None] + x[None]
```

## Writing Kernels
The triton kernel should have the following functionality

    
if __name__ == "__main__":
    _test_puzzle(outer_vector_add_kernel, outer_vector_add, nelem={"N0": 32, "N1": 32})
