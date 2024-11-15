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
    # Calculate program ID and batch index 
    pid = tl.program_id(0)
    batch_idx = pid * B0

    # Calculate offsets 
    x_offset = batch_idx * T
    z_offset = pid

    # Allocate local accumulator
    acc = tl.zeros([B0], dtype=tl.float32)

    # Loop through sub-blocks of the batch 
    for start in range(0, T, B1):
        # Load mask in case of incomplete block
        mask = (start + tl.arange(0, B1)) < T

        # Load and accumulate 
        block = tl.load(x_ptr + x_offset + start + tl.arange(0, B1), mask=mask)
        acc += block

    # Store final sum 
    tl.store(z_ptr + z_offset, tl.sum(acc))

    
if __name__ == "__main__":
    _test_puzzle(long_sum_kernel, long_sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})
