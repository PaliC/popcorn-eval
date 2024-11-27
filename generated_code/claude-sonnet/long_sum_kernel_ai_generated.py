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

# Helper to compute the next power of 2 for rounding
@triton.jit
def next_power_of_2(v):
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v += 1
    return v

# Kernel to sum batches of numbers, processing B1 elements at a time
@triton.jit
def long_sum_kernel(
    x_ptr,  # Input pointer [B0, N0, T]
    z_ptr,  # Output pointer [B0, N0]
    N0,     # Number of items in batch
    T,      # Length of each item
    B0: tl.constexpr,  # Block size for batches
    B1: tl.constexpr,  # Block size for processing elements
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute batch/item indices
    batch_idx = pid // N0  
    item_idx = pid % N0

    # Bounds check
    if batch_idx >= B0:
        return

    # Initialize accumulator
    acc = 0.0

    # Process T elements B1 at a time
    for i in range(0, T, B1):
        # Load B1 elements
        block_start = batch_idx * N0 * T + item_idx * T + i
        mask = tl.arange(0, B1) < (T - i)
        x = tl.load(x_ptr + block_start, mask=mask, other=0)
        
        # Add to accumulator
        acc += tl.sum(x, axis=0)

    # Store final sum
    out_idx = batch_idx * N0 + item_idx
    tl.store(z_ptr + out_idx, acc)

    
if __name__ == "__main__":
    _test_puzzle(long_sum_kernel, long_sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})
