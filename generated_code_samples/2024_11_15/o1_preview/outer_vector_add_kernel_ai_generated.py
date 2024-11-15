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
def outer_vector_add_kernel(x_ptr, y_ptr, z_ptr, N0, B0: tl.constexpr):
    # Create indices for x and y
    i = tl.arange(0, B0)
    j = tl.arange(0, B0)
    # Load x and y values
    x = tl.load(x_ptr + i)
    y = tl.load(y_ptr + j)
    # Compute outer sum
    z = x[:, None] + y[None, :]
    # Compute flat indices for z
    z_idx = i[:, None] * N0 + j[None, :]
    # Store the result
    tl.store(z_ptr + z_idx, z)

    
if __name__ == "__main__":
    _test_puzzle(outer_vector_add_kernel, outer_vector_add, nelem={"N0": 32, "N1": 32})
