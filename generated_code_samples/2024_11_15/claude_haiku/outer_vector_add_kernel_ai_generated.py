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
    Performs outer vector addition: z[i, j] = x[i] + y[j]
    """
    # Get current row and column indices
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    # Load x and y values for current block
    x_val = tl.load(x_ptr + row_idx * B0 + tl.arange(0, B0))
    y_val = tl.load(y_ptr + col_idx * B0 + tl.arange(0, B0))

    # Compute elementwise addition 
    z_val = x_val[:, None] + y_val[None, :]

    # Store result in output matrix
    z_ptr += row_idx * B0 * N0 + col_idx * B0
    tl.store(z_ptr, z_val)

    
if __name__ == "__main__":
    _test_puzzle(outer_vector_add_kernel, outer_vector_add, nelem={"N0": 32, "N1": 32})
