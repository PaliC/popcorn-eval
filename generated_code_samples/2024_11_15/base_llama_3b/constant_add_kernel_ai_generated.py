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

N0 = 5
```

  The kernel should output the result of the addition operation to the `z_ptr` argument. The kernel should be written in a reusable way. It should take a single argument `N0` which should be constant. If the user is only passing in a single argument, the kernel should be able to handle it. The kernel should also accept a batch size as an additional argument. If the user only passes in a single argument, the kernel should be able to handle it. It should print a warning message to the console if the user passes in an incorrect number of arguments.

  The kernel should have the following function signature:
```python
def constant_add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
```

  The kernel should be written in the triton DSL. The kernel should perform an elementwise addition operation and add a constant value of 10 to each element of the input vector. The kernel uses the following constant values:
```python
N0 = 5

    
if __name__ == "__main__":
    _test_puzzle(constant_add_kernel, constant_add, nelem={"N0": 32})
