import triton
import triton.language as tl
import torch
import inspect
from _helper_functions import _test_puzzle
from jaxtyping import Float32, Int32
from torch import Tensor

# taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles

def constant_add(x: Float32[Tensor, "32"]) -> Float32[Tensor, "32"]:
    return x + 10

@triton.jit
def constant_add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    """
    Elementwise addition of constant 10 to input vector
    
    Args:
        x_ptr: Pointer to input vector 
        z_ptr: Pointer to output vector
        N0: Total number of elements 
        B0: Block size for parallel processing
    """
    # Compute global thread index 
    pid = tl.program_id(axis=0)
    
    # Calculate offset for current block
    block_start = pid * B0
    
    # Create offset vector for loading elements
    offsets = block_start + tl.arange(0, B0)
    
    # Create mask to handle boundary conditions
    mask = offsets < N0
    
    # Load input elements 
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Add constant 10 to input elements
    z = x + 10
    
    # Store result 
    tl.store(z_ptr + offsets, z, mask=mask)

    
if __name__ == "__main__":
    _test_puzzle(constant_add_kernel, constant_add, nelem={"N0": 32})
