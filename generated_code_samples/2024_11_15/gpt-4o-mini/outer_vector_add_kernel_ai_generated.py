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
    # Get the indices for the current block
    row = tl.program_id(0)  # Row index
    col = tl.arange(0, B1)  # Column index
    
    # Load from global memory
    x_val = tl.load(x_ptr + row)  # Load row from x
    y_val = tl.load(y_ptr + col)  # Load column from y
    
    # Perform elementwise addition
    z_val = x_val + y_val
    
    # Write the result to global memory
    tl.store(z_ptr + row * N1 + col, z_val)

    
if __name__ == "__main__":
    _test_puzzle(outer_vector_add_kernel, outer_vector_add, nelem={"N0": 32, "N1": 32})
