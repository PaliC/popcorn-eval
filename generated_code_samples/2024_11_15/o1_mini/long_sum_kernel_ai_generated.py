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

import torch
import triton
import triton.language as tl

@triton.jit
def long_sum_kernel(x_ptr, z_ptr, N0, T, B0: tl.constexpr, B1: tl.constexpr):
    # Calculate program ID
    pid = tl.program_id(0)
    # Calculate the starting batch index
    block_start = pid * B0
    # Initialize accumulator
    acc = tl.zeros((B1,), dtype=tl.float32)
    # Iterate over the T dimension in steps of B1
    for offset in range(0, T, B1):
        # Compute the current position
        pos = block_start + tl.arange(0, B0)[:, None] * T + offset + tl.arange(0, B1)[None, :]
        # Load a block of data
        x = tl.load(x_ptr + pos)
        # Sum along the batch dimension
        acc += tl.sum(x, axis=0)
    # Store the result
    z_ptr[pid * B1 + tl.arange(0, B1)] = acc

    
if __name__ == "__main__":
    _test_puzzle(long_sum_kernel, long_sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})
