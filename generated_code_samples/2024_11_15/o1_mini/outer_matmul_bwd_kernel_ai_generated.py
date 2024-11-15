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

import torch
import triton
import triton.language as tl

@triton.jit
def outer_matmul_bwd_kernel(
    x_ptr, y_ptr, dz_ptr, dx_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    # Calculate grid dimensions
    grid_n0 = (N0 + B0 - 1) // B0
    grid_n1 = (N1 + B1 - 1) // B1
    grid = grid_n0 * grid_n1
    pid = tl.program_id(0)
    block_id0 = pid // grid_n1
    block_id1 = pid % grid_n1

    # Compute block start indices
    offs0 = block_id0 * B0 + tl.arange(0, B0)
    offs1 = block_id1 * B1 + tl.arange(0, B1)

    # Create masks for valid indices
    mask0 = offs0 < N0
    mask1 = offs1 < N1

    # Load y values
    y = tl.load(y_ptr + offs1, mask=mask1, other=0.0)

    # Compute linear indices for dz
    dz_indices = offs0[:, None] * N1 + offs1[None, :]

    # Load dz values
    dz = tl.load(dz_ptr + dz_indices, mask=mask0[:, None] & mask1[None, :], other=0.0)

    # Multiply dz and y
    prod = dz * y

    # Sum over B1 dimension
    acc = tl.sum(prod, axis=1)

    # Iterate over B0 to perform atomic adds
    for i in range(B0):
        idx = offs0[i]
        m = mask0[i]
        # Atomic add to dx
        tl.atomic_add(dx_ptr + idx, acc[i], mask=m)


if __name__ == "__main__":
    _test_puzzle(
        outer_matmul_bwd_kernel, outer_matmul_bwd_spec, nelem={"N0": 100, "N1": 90}
    )
