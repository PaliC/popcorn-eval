import triton
import triton.language as tl
import torch
import inspect
from _helper_functions import _test_puzzle

# taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles

def constant_add(x: Float32[Tensor, "32"]) -> Float32[Tensor, "32"]:
    return x + 10

@triton.jit
def constant_add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    # Compute the current element's index in the global memory
    pid = tl.program_id(axis=0)
    
    # Calculate the offset for the current block
    block_start = pid * B0
    
    # Create an offset array for loading elements within the block
    offs = block_start + tl.arange(0, B0)
    
    # Mask to handle cases where the block might go beyond array bounds
    mask = offs < N0
    
    # Load input elements
    x = tl.load(x_ptr + offs, mask=mask)
    
    # Add constant value to each element
    z = x + 10
    
    # Store result back to output pointer
    tl.store(z_ptr + offs, z, mask=mask)

    
if __name__ == "__main__":
    _test_puzzle(constant_add, constant_add_kernel, nelem={"N0": 32})
