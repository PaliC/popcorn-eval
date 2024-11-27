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
    # Get program ID for row and column
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    # Create masks for out of bounds handling
    x_mask = pid0 < N0
    y_mask = pid1 < N0

    # Load x and y values 
    x_val = tl.load(x_ptr + pid0, mask=x_mask)
    y_val = tl.load(y_ptr + pid1, mask=y_mask)

    # Compute output value 
    z_val = x_val + y_val

    # Store result in output pointer 
    z_offset = pid0 * N0 + pid1
    tl.store(z_ptr + z_offset, z_val, mask=(x_mask & y_mask))

    
if __name__ == "__main__":
    _test_puzzle(outer_vector_add_kernel, outer_vector_add, nelem={"N0": 32, "N1": 32})
