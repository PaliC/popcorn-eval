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
    # Define block offsets for x and y
    b0 = tl.program_id(0) * B0 + tl.arange(0, B0)
    b1 = tl.program_id(1) * B1 + tl.arange(0, B1)

    # Load dz values for the corresponding output indices
    dz = tl.load(dz_ptr + b0[:, None])

    # Load x and y values
    x = tl.load(x_ptr + b0[:, None])
    y = tl.load(y_ptr + b1[None, :])

    # Compute outer product and accumulate into dx
    dx = tl.dot(dz, y)
    
    # Store the result in the output tensor
    tl.store(dx_ptr + b0[:, None], dx)


if __name__ == "__main__":
    _test_puzzle(
        outer_matmul_bwd_kernel, outer_matmul_bwd_spec, nelem={"N0": 100, "N1": 90}
    )
