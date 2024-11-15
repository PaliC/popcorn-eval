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
    # Compute pid for rows and columns
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    
    # Calculate row and column offsets
    row_start = pid0 * B0
    col_start = pid1 * B1

    # Create masks to handle boundary conditions
    row_mask = (row_start + tl.arange(0, B0)) < N0
    col_mask = (col_start + tl.arange(0, B1)) < N1

    # Load x vector block
    x_block = tl.load(x_ptr + row_start + tl.arange(0, B0), mask=row_mask)

    # Load y vector block 
    y_block = tl.load(y_ptr + col_start + tl.arange(0, B1), mask=col_mask)

    # Load gradient of output block
    dz_block = tl.load(dz_ptr + row_start + tl.arange(0, B0), mask=row_mask)

    # Compute gradient for x
    dx_block = tl.sum(dz_block[:, None] * y_block[None, :], axis=1)

    # Store gradient 
    tl.store(dx_ptr + row_start + tl.arange(0, B0), dx_block, mask=row_mask)


if __name__ == "__main__":
    _test_puzzle(
        outer_matmul_bwd_kernel, outer_matmul_bwd_spec, nelem={"N0": 100, "N1": 90}
    )
