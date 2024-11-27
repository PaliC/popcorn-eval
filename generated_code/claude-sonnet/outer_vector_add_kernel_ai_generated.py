import triton
import triton.language as tl
import torch
import inspect
from _helper_functions import _test_puzzle
import jaxtyping
from jaxtyping import Float32, Int32
from torch import Tensor

# taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles

def outer_vector_add(x: Float32[Tensor, "32"], y: Float32[Tensor, "32"]) -> Float32[Tensor, "32 32"]:
    return x[None, :] + y[:, None]

@triton.jit
def outer_vector_add_kernel(x_ptr, y_ptr, z_ptr, N0, B0: tl.constexpr):
    """Kernel for calculating outer vector addition z[i,j] = x[i] + y[j]
    
    Args:
        x_ptr: Pointer to input vector x
        y_ptr: Pointer to input vector y
        z_ptr: Pointer to output matrix z 
        N0: Length of vectors x and y
        B0: Block size (same as N0)
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Load x
    x_block_ptr = x_ptr + pid * B0
    x = tl.load(x_block_ptr + tl.arange(0, B0))
    
    # Load y
    y_block_ptr = y_ptr + pid * B0 
    y = tl.load(y_block_ptr + tl.arange(0, B0))
    
    # Calculate output indices for 2D output
    row_idx = pid * B0
    offsets = tl.arange(0, B0)
    
    # Outer addition - add x to each element of y
    for j in range(B0):
        out_offset = row_idx * N0 + j * N0 + offsets
        tl.store(z_ptr + out_offset, x + y[j])

    
if __name__ == "__main__":
    _test_puzzle(outer_vector_add_kernel, outer_vector_add, nelem={"N0": 32, "N1": 32})
