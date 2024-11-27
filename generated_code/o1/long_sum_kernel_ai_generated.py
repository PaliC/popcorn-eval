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
    # Compute batch indices
    batch_start = tl.program_id(0) * B0
    batch_offsets = batch_start + tl.arange(0, B0)
    batch_mask = batch_offsets < N0
    # Initialize sums
    sums = tl.zeros([B0], dtype=tl.float32)
    # Loop over T dimension in steps of B1
    for t_offset in range(0, T, B1):
        t_offsets = t_offset + tl.arange(0, B1)
        t_mask = t_offsets < T
        # Compute memory addresses
        x_ptrs = x_ptr + batch_offsets[:, None] * T + t_offsets[None, :]
        # Load data with masking
        x = tl.load(x_ptrs, mask=batch_mask[:, None] & t_mask[None, :], other=0.0)
        # Accumulate sums
        sums += tl.sum(x, axis=1)
    # Store results
    z_ptrs = z_ptr + batch_offsets
    tl.store(z_ptrs, sums, mask=batch_mask)

    
if __name__ == "__main__":
    _test_puzzle(long_sum_kernel, long_sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})
