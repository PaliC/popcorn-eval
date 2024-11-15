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
    # Compute global thread index
    pid = tl.program_id(0)
    # Calculate row and column based on block size
    row = pid // B0
    col = pid % B0
    # Load y[row] and x[col]
    y = tl.load(y_ptr + row)
    x = tl.load(x_ptr + col)
    # Compute addition
    z = y + x
    # Store the result
    tl.store(z_ptr + pid, z)

    
if __name__ == "__main__":
    _test_puzzle(outer_vector_add_kernel, outer_vector_add, nelem={"N0": 32, "N1": 32})
