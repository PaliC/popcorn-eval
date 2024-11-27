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
    # Program IDs for each block
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    # Compute the start indices of this block
    offs_x = pid_x * B0 + tl.arange(0, B0)
    offs_y = pid_y * B0 + tl.arange(0, B0)
    # Masks for bounds checking
    mask_x = offs_x < N0
    mask_y = offs_y < N0
    # Load x and y values for the current block
    x = tl.load(x_ptr + offs_x, mask=mask_x, other=0.0)
    y = tl.load(y_ptr + offs_y, mask=mask_y, other=0.0)
    # Expand x and y to 2D tensors
    x = x[None, :]  # Shape [1, B0]
    y = y[:, None]  # Shape [B0, 1]
    # Compute outer addition
    z = x + y  # Shape [B0, B0]
    # Compute linear indices for storing the result
    linear_row = offs_y[:, None] * N0  # Shape [B0, 1]
    linear_col = offs_x[None, :]       # Shape [1, B0]
    linear_indices = linear_row + linear_col  # Shape [B0, B0]
    # Combine masks for storing
    mask = mask_y[:, None] & mask_x[None, :]  # Shape [B0, B0]
    # Store the result
    tl.store(z_ptr + linear_indices, z, mask=mask)

    
if __name__ == "__main__":
    _test_puzzle(outer_vector_add_kernel, outer_vector_add, nelem={"N0": 32, "N1": 32})
