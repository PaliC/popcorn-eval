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
def long_sum_kernel(x_ptr, z_ptr, N0, T, B0: tl.constexpr, B1: tl.constexpr):
    # Program ID for batch dimension
    pid0 = tl.program_id(0)
    # Compute batch indices
    off_n = pid0 * B0 + tl.arange(0, B0)
    batch_mask = off_n < N0
    # Initialize sums
    sums = tl.zeros([B0], dtype=tl.float32)
    # Loop over elements in T dimension
    for t_start in range(0, T, B1):
        # Compute element indices
        off_t = t_start + tl.arange(0, B1)
        t_mask = off_t < T
        # Compute memory offsets
        ptrs = x_ptr + off_n[:, None] * T + off_t[None, :]
        # Combine masks
        mask = batch_mask[:, None] & t_mask[None, :]
        # Load input data
        x = tl.load(ptrs, mask=mask, other=0.0)
        # Accumulate sums
        sums += tl.sum(x, axis=1)
    # Store results
    z_ptrs = z_ptr + off_n
    tl.store(z_ptrs, sums, mask=batch_mask)

    
if __name__ == "__main__":
    _test_puzzle(long_sum_kernel, long_sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})
