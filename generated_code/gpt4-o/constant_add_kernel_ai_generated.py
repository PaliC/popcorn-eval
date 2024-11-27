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

def constant_add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    # Calculate the block index
    block_id = tl.program_id(0)
    
    # Calculate the position of the first element in this block
    offsets = block_id * B0 + tl.arange(0, B0)
    
    # Mask to handle out-of-bounds for the last block
    mask = offsets < N0
    
    # Load data from x_ptr
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Add constant 10 to each element
    z = x + 10.0
    
    # Store the result back to z_ptr
    tl.store(z_ptr + offsets, z, mask=mask)

    
if __name__ == "__main__":
    _test_puzzle(constant_add_kernel, constant_add, nelem={"N0": 32})
