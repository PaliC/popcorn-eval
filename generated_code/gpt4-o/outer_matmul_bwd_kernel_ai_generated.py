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

def outer_matmul_bwd_kernel(
    x_ptr, y_ptr, dz_ptr, dx_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    # Offsets for x and dz
    bx = tl.program_id(0) * B0
    idx_x = bx + tl.arange(0, B0)
    mask_x = idx_x < N0

    # Compute gradient dx: dx[i] = dz[i] * sum(y)
    # Load dz segment
    dz = tl.load(dz_ptr + idx_x, mask=mask_x, other=0.0)

    # Initialize dx to zero
    dx = tl.zeros([B0], dtype=tl.float32)

    for by in range(0, N1, B1):
        # Compute offsets for y
        idx_y = by + tl.arange(0, B1)
        mask_y = idx_y < N1

        # Load y segment
        y = tl.load(y_ptr + idx_y, mask=mask_y, other=0.0)

        # Compute sum(y)
        sum_y = tl.sum(y, axis=0)

        # Accumulate dx
        dx += dz * sum_y

    # Store the result back to memory
    tl.store(dx_ptr + idx_x, dx, mask=mask_x)


if __name__ == "__main__":
    _test_puzzle(
        outer_matmul_bwd_kernel, outer_matmul_bwd_spec, nelem={"N0": 100, "N1": 90}
    )
