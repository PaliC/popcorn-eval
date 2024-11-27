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
    """Backward pass for outer product matmul: dz * y -> dx"""
    # Program id in blocks
    pid0 = tl.program_id(0) 
    pid1 = tl.program_id(1)

    # Initialize offsets for this program block 
    offs0 = pid0 * B0 + tl.arange(0, B0)
    offs1 = pid1 * B1 + tl.arange(0, B1)

    # Create masks for bounds checking
    mask0 = offs0 < N0
    mask1 = offs1 < N1

    # Load inputs for this block
    x = tl.load(x_ptr + offs0, mask=mask0)
    y = tl.load(y_ptr + offs1, mask=mask1)
    dz = tl.load(dz_ptr + offs0, mask=mask0)

    # Compute outer product grad
    dx = dz[:, None] * y[None, :]
    
    # Accumulate gradients 
    tl.store(dx_ptr + offs0 * N1 + offs1, dx, mask=mask0[:, None] & mask1[None, :])


if __name__ == "__main__":
    _test_puzzle(
        outer_matmul_bwd_kernel, outer_matmul_bwd_spec, nelem={"N0": 100, "N1": 90}
    )
