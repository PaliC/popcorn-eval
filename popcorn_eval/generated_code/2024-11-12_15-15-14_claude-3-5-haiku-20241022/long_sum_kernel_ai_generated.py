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
    # Batch index for current block
    pid = tl.program_id(0)
    
    # Ensure we don't exceed total batches
    if pid < N0:
        # Pointer to start of current batch
        x_batch_ptr = x_ptr + pid * T
        
        # Accumulator for batch sum
        batch_sum = tl.zeros(B0, dtype=tl.float32)
        
        # Process batch in chunks of B1 elements
        for offs in range(0, T, B1):
            # Bounds check
            mask = (offs + tl.arange(0, B1)) < T
            
            # Load elements and mask
            x_vals = tl.load(x_batch_ptr + offs + tl.arange(0, B1), mask=mask)
            
            # Accumulate sum
            batch_sum += x_vals
        
        # Store final sum for this batch
        z_ptr[pid] = tl.sum(batch_sum)

    
if __name__ == "__main__":
    _test_puzzle(long_sum_kernel, long_sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})
