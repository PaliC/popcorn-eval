import inspect

import torch
import triton
import triton.language as tl
from _helper_functions import _test_puzzle
from jaxtyping import Float32, Int32
from torch import Tensor

# taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles


def outer_matmul_bwd_spec(
    x: Float32[Tensor, "90 100"],
    y: Float32[Tensor, "90"],
    dz: Float32[Tensor, "90 100"],
) -> Float32[Tensor, "90 100"]:
    x = x.clone()
    y = y.clone()
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = torch.relu(x * y[:, None])
    z.backward(dz)
    dx = x.grad
    return dx

@triton.jit
def outer_matmul_bwd_kernel(
    x_ptr, y_ptr, dz_ptr, dx_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    # Program ID for dimension 0 
    pid0 = tl.program_id(0)
    
    # Calculate offsets for x and dz blocks
    x_offset = pid0 * B0
    dz_offset = pid0 * B0
    
    # Mask to handle boundary conditions
    x_mask = (x_offset + tl.arange(0, B0)) < N0
    
    # Load x block with boundary handling
    x_block = tl.load(x_ptr + x_offset + tl.arange(0, B0), mask=x_mask)
    
    # Program ID for dimension 1
    pid1 = tl.program_id(1)
    
    # Calculate offsets for y block
    y_offset = pid1 * B1
    
    # Mask to handle boundary conditions for y
    y_mask = (y_offset + tl.arange(0, B1)) < N1
    
    # Load y block with boundary handling
    y_block = tl.load(y_ptr + y_offset + tl.arange(0, B1), mask=y_mask)
    
    # Load dz block 
    dz_block = tl.load(dz_ptr + x_offset, mask=x_mask)
    
    # Compute outer product and accumulate
    # Broadcast dz across rows and y across columns 
    dx_block = dz_block[:, None] * y_block[None, :]
    
    # Store results with boundary handling
    tl.store(
        dx_ptr + x_offset * N1 + y_offset, 
        dx_block, 
        mask=(x_mask[:, None] & y_mask[None, :])
    )


if __name__ == "__main__":
    _test_puzzle(
        outer_matmul_bwd_kernel, outer_matmul_bwd_spec, nelem={"N0": 100, "N1": 90}
    )
