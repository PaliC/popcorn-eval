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
    """
    Computes outer product of x and y vectors with ReLU activation
    """
    # Current block indices for x and y 
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    # Calculate starting offsets for x and y blocks 
    x_block_start = pid0 * B0
    y_block_start = pid1 * B1

    # Create mask and boundary checks
    x_mask = (x_block_start + tl.arange(0, B0) < N0)
    y_mask = (y_block_start + tl.arange(0, B1) < N1)

    # Load x and y blocks
    x_block = tl.load(x_ptr + x_block_start + tl.arange(0, B0), mask=x_mask)
    y_block = tl.load(y_ptr + y_block_start + tl.arange(0, B1), mask=y_mask)

    # Compute outer product block with ReLU
    z_block = tl.maximum(x_block[:, None] * y_block[None, :], 0)

    # Store result block 
    z_start = pid0 * B0 + pid1 * B1 * N0
    tl.store(z_ptr + z_start + tl.arange(0, B0)[:, None] * N0 + tl.arange(0, B1)[None, :], 
             z_block, mask=x_mask[:, None] & y_mask[None, :])

    
if __name__ == "__main__":
    _test_puzzle(outer_matmul_kernel,outer_matmul_spec, nelem={"N0": 100, "N1": 90})
