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
    Elementwise addition of a constant (10) to input vector
    
    :param x_ptr: Pointer to input vector 
    :param z_ptr: Pointer to output vector
    :param N0: Vector length
    :param B0: Block size (compilation-time constant)
    """
    # Get current program ID and current offset 
    pid = tl.program_id(0)
    offs = pid * B0 + tl.arange(0, B0)
    
    # Boundary check to prevent out-of-bounds access
    mask = offs < N0
    
    # Load elements and add constant 
    x = tl.load(x_ptr + offs, mask=mask)
    z = x + 10
    
    # Store result
    tl.store(z_ptr + offs, z, mask=mask)

    
if __name__ == "__main__":
    _test_puzzle(constant_add_kernel, constant_add, nelem={"N0": 32})
