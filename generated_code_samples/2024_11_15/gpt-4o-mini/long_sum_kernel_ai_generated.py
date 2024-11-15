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
    # Get the batch index
    batch_idx = tl.program_id(0)
    if batch_idx >= N0:
        return  # Out of bounds check

    # Compute base address for the current batch
    offset = batch_idx * T
    x_batch = x_ptr + offset

    # Initialize sum for the current batch
    sum_val = tl.zeros((1,), dtype=tl.float32)

    # Process B1 elements at a time
    for i in range(0, T, B1):
        # Load elements into a temporary vector
        x_data = tl.load(x_batch + i, num=B1, mask=i + tl.arange(0, B1) < T)
        sum_val += tl.sum(x_data)

    # Store the result
    z_ptr[batch_idx] = sum_val

    
if __name__ == "__main__":
    _test_puzzle(long_sum_kernel, long_sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})
