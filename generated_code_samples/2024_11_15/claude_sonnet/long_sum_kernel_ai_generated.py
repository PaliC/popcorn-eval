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
    """Sum a batch of vectors along inner dimension
    Args:
        x_ptr: Input tensor pointer [N0, T] 
        z_ptr: Output tensor pointer [N0]
        N0: Batch dimension
        T: Inner dimension length
        B0: Block size for batch dimension
        B1: Block size for inner dimension
    """
    # Program ID for batch dimension
    pid0 = tl.program_id(0)
    
    # Compute starting index for this block's batch
    batch_start = pid0 * B0
    batch_offsets = batch_start + tl.arange(0, B0)
    
    # Mask out of bounds batch indices
    batch_mask = batch_offsets < N0
    
    # Initialize accumulator for each batch element
    acc = tl.zeros([B0], dtype=tl.float32)
    
    # Process inner dimension in blocks of size B1
    for t in range(0, T, B1):
        # Get inner dimension offsets and mask
        t_offs = t + tl.arange(0, B1)
        t_mask = t_offs < T
        
        # Combined mask for both dimensions
        mask = batch_mask[:, None] & t_mask[None, :]
        
        # Load and accumulate
        # x has shape [N0, T] so offset is batch_idx * T + t_idx
        offs = batch_offsets[:, None] * T + t_offs[None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        acc += tl.sum(x, axis=1)
    
    # Store result
    tl.store(z_ptr + batch_offsets, acc, mask=batch_mask)

    
if __name__ == "__main__":
    _test_puzzle(long_sum_kernel, long_sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})
