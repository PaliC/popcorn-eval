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

def outer_matmul_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    offset_m = pid0 * B1
    offset_n = pid1 * B0
    indices_m = offset_m + tl.arange(0, B1)
    indices_n = offset_n + tl.arange(0, B0)
    x_vals = tl.load(x_ptr + indices_n, mask=indices_n < N0, other=0.0)
    y_vals = tl.load(y_ptr + indices_m, mask=indices_m < N1, other=0.0)
    out = y_vals[:, None] * x_vals[None, :]
    out = tl.relu(out)
    offsets = indices_m[:, None] * N0 + indices_n[None, :]
    mask = (indices_m[:, None] < N1) & (indices_n[None, :] < N0)
    tl.store(z_ptr + offsets, out, mask=mask)

    
if __name__ == "__main__":
    _test_puzzle(outer_matmul_kernel,outer_matmul_spec, nelem={"N0": 100, "N1": 90})
