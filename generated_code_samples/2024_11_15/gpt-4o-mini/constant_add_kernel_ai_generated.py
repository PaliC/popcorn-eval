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
def constant_add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    # Get the unique index for each thread
    pid = tl.program_id(0)
    stride = B0 * N0
    offsets = pid * B0 + tl.arange(0, B0)

    # Load input elements
    x = tl.load(x_ptr + offsets)

    # Add constant value of 10
    z = x + 10

    # Store result in output pointer
    tl.store(z_ptr + offsets, z)

    
if __name__ == "__main__":
    _test_puzzle(constant_add_kernel, constant_add, nelem={"N0": 32})
