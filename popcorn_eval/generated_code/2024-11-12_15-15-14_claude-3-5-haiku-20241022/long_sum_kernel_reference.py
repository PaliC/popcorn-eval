import triton
import triton.language as tl
import torch
import inspect
from _helper_functions import _test_puzzle
from jaxtyping import Float32, Int32
from torch import Tensor

# taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles

def long_sum_spec(x: Float32[Tensor, "4 200"]) -> Float32[Tensor, "4"]:
    return x.sum(1)

@triton.jit
def long_sum_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    pid_0 = tl.program_id(0)

    range_0 = pid_0 * B0 + tl.arange(0, B0)
    mask_0 = range_0 < N0
    _sum = tl.arange(0, B0) * 0.0

    for offset_col in tl.static_range(0, T, B1):
        range_1 = offset_col + tl.arange(0, B1)
        range_block = range_0[:, None] * T + range_1[None, :]  # (B0, B1)
        mask_block = mask_0[:, None] & (range_1[None, :] < T)

        x_block = tl.load(x_ptr + range_block, mask=mask_block, other=0.0)
        _sum += tl.sum(x_block, axis=1)

    tl.store(z_ptr + range_0, _sum, mask=mask_0)

    return

    
if __name__ == "__main__":
    _test_puzzle(long_sum_kernel, long_sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})
