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
    # Program ID and block ID for parallelization
    pid = tl.program_id(0)
    
    # Calculate memory offsets
    offsets_x = tl.arange(0, B0)
    offsets_y = tl.arange(0, B0) + pid * B0

    # Load elements from x and y
    x_elements = tl.load(x_ptr + offsets_x)
    y_element = tl.load(y_ptr + offsets_y)

    # Compute outer addition
    result = x_elements + y_element[:, None]

    # Store result back to z
    tl.store(z_ptr + offsets_y[:, None] * N0 + offsets_x, result)

    
if __name__ == "__main__":
    _test_puzzle(outer_vector_add_kernel, outer_vector_add, nelem={"N0": 32, "N1": 32})
