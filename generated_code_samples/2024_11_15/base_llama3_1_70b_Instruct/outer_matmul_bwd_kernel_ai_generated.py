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
    x_ptr: tl.constexpr
    y_ptr: tl.constexpr
    dz_ptr: tl.constexpr
    dx_ptr: tl.constexpr
    N0: tl.constexpr
    N1: tl.constexpr
    B0: tl.constexpr
    B1: tl.constexpr
):
    # |x| = |y| * |z| where |z| is a matrix of shape |N0| x |N1|
    # |dz| = |dx| * |dy| + |dy| * |dx|
    # |dx| = |dz| / |N0|
    # |dy| = |dz| / |N1|


if __name__ == "__main__":
    _test_puzzle(
        outer_matmul_bwd_kernel, outer_matmul_bwd_spec, nelem={"N0": 100, "N1": 90}
    )
