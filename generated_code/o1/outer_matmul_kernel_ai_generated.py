import triton
import triton.language as tl
import torch
import inspect
from _helper_functions import _test_puzzle
from jaxtyping import Float32, Int32
from torch import Tensor

# taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles

def outer_matmul_spec(x: Float32[Tensor, "100"], y: Float32[Tensor, "90"]) -> Float32[Tensor, "90 100"]:
    return torch.relu(x[None, :] * y[:, None])

@triton.jit
def outer_matmul_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    # Program IDs for the two axes
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    # Offsets for x and y
    offs_x = pid0 * B0 + tl.arange(0, B0)
    offs_y = pid1 * B1 + tl.arange(0, B1)
    # Masks for x and y
    mask_x = offs_x < N0
    mask_y = offs_y < N1
    # Load x and y
    x = tl.load(x_ptr + offs_x, mask=mask_x, other=0.0)
    y = tl.load(y_ptr + offs_y, mask=mask_y, other=0.0)
    # Compute outer product and apply ReLU
    z = tl.outer(x, y)
    z = tl.maximum(z, 0.0)
    # Offsets for z
    z_offs0 = pid0 * B0 + tl.arange(0, B0)
    z_offs1 = pid1 * B1 + tl.arange(0, B1)
    # Create a 2D grid of offsets
    z_idx0 = z_offs0[:, None]
    z_idx1 = z_offs1[None, :]
    z_offsets = z_idx0 * N1 + z_idx1
    # Mask for z
    mask_z = (z_idx0 < N0) & (z_idx1 < N1)
    # Store result
    tl.store(z_ptr + z_offsets, z, mask=mask_z)

    
if __name__ == "__main__":
    _test_puzzle(outer_matmul_kernel,outer_matmul_spec, nelem={"N0": 100, "N1": 90})
