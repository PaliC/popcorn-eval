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
    # Get block indices for rows and columns
    block_i = tl.program_id(0)
    block_j = tl.program_id(1)
    
    # Calculate offsets for x and y
    offsets_i = block_i * B0 + tl.arange(0, B0)
    offsets_j = block_j * B1 + tl.arange(0, B1)
    
    # Create masks to handle boundary conditions
    mask_i = offsets_i < N0
    mask_j = offsets_j < N1
    
    # Load x and y vectors with masking
    x = tl.load(x_ptr + offsets_i, mask=mask_i, other=0.0)
    y = tl.load(y_ptr + offsets_j, mask=mask_j, other=0.0)
    
    # Reshape x and y for outer product
    x = x[:, None]  # Shape: (B0, 1)
    y = y[None, :]  # Shape: (1, B1)
    
    # Compute outer product and apply ReLU
    z = tl.maximum(x * y, 0.0)
    
    # Compute absolute indices for z
    i = offsets_i[:, None]
    j = offsets_j[None, :]
    
    # Create mask for storing results
    mask = (i < N0) & (j < N1)
    
    # Store the computed values to z_ptr
    tl.store(z_ptr + i * N1 + j, z, mask=mask)

    
if __name__ == "__main__":
    _test_puzzle(outer_matmul_kernel,outer_matmul_spec, nelem={"N0": 100, "N1": 90})
