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
    # Calculate the batch index
    batch_idx = tl.program_id(0)

    # Calculate the offsets for batch start and end
    start_idx = batch_idx * T
    end_idx = start_idx + T

    # Initialize the sum for this batch to 0
    batch_sum = tl.zeros((), dtype=tl.float32)

    # Iterate over the elements in the batch in increments of B1
    for i in range(start_idx, end_idx, B1):
        # Load a block of B1 elements from global memory
        offsets = i + tl.arange(0, B1)
        elements = tl.load(x_ptr + offsets, mask=offsets < N0 * T)
        
        # Sum the loaded elements
        batch_sum += tl.sum(elements, axis=0)
    
    # Write the batch sum to the output memory
    tl.store(z_ptr + batch_idx, batch_sum)

    
if __name__ == "__main__":
    _test_puzzle(long_sum_kernel, long_sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})
