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
    """
    Computes outer vector addition by broadcasting x and y
    
    Args:
        x_ptr: Pointer to first input vector 
        y_ptr: Pointer to second input vector
        z_ptr: Pointer to output matrix 
        N0: Length of x vector
        B0: Block size (same as N0)
    """
    # Get current program's row and column indices 
    pid_m = tl.program_id(0)   # row index
    pid_n = tl.program_id(1)   # column index

    # Load x vector element for current row
    x_elem = tl.load(x_ptr + pid_m, mask=pid_m < N0)
    
    # Load y vector element for current column 
    y_elem = tl.load(y_ptr + pid_n, mask=pid_n < N0)
    
    # Compute z element as outer product addition
    z_elem = x_elem + y_elem
    
    # Store result in output matrix
    z_offset = pid_m * N0 + pid_n
    tl.store(z_ptr + z_offset, z_elem, mask=(pid_m < N0) & (pid_n < N0))

    
if __name__ == "__main__":
    _test_puzzle(outer_vector_add_kernel, outer_vector_add, nelem={"N0": 32})
