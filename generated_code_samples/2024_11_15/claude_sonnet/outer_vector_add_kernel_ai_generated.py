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
    """Kernel to compute outer product addition z[i,j] = x[i] + y[j]
    Args:
        x_ptr: Pointer to first input vector x
        y_ptr: Pointer to second input vector y  
        z_ptr: Pointer to output matrix z
        N0: Length of input vectors x and y
        B0: Block size (same as N0)
    """
    # Program id is row index
    pid = tl.program_id(0)

    # Load full vectors since block size = vector length
    x = tl.load(x_ptr + tl.arange(0, B0))
    y = tl.load(y_ptr + tl.arange(0, B0)) 

    # Compute one row of output
    row_idx = pid * B0
    offs = tl.arange(0, B0)
    z = x[pid] + y[offs]

    # Store result row
    z_ptrs = z_ptr + row_idx + offs
    tl.store(z_ptrs, z)

    
if __name__ == "__main__":
    _test_puzzle(outer_vector_add_kernel, outer_vector_add, nelem={"N0": 32, "N1": 32})
