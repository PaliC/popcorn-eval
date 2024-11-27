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
    # Compute batch index and thread index within batch
    batch_idx = tl.program_id(0)
    thread_idx = tl.arange(0, B1)
    
    # Check if batch is within range 
    if batch_idx < N0:
        # Compute base pointer for input and output
        x_base = x_ptr + batch_idx * T + thread_idx
        z_base = z_ptr + batch_idx
        
        # Initialize accumulator 
        acc = tl.zeros([B1], dtype=tl.float32)
        
        # Iterate and accumulate in chunks of B1
        for i in range(0, T, B1):
            # Load elements 
            mask = (i + thread_idx) < T
            x_val = tl.load(x_base, mask=mask)
            
            # Accumulate 
            acc += x_val
            
            # Move base pointer
            x_base += B1
        
        # Reduce accumulated values 
        z_val = tl.sum(acc)
        
        # Store result
        tl.store(z_base, z_val)

    
if __name__ == "__main__":
    _test_puzzle(long_sum_kernel, long_sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})
