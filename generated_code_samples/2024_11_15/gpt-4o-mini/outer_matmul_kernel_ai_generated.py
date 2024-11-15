import triton
import triton.language as tl
import torch
import inspect
from _helper_functions import _test_puzzle
from jaxtyping import Float32, Int32
from torch import Tensor

# taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles

def outer_matmul_spec(x: Float32[Tensor, "100"], y: Float32[Tensor, "90"]) -> Float32[Tensor, "90 100"]:
    return torch.relu(x[None, :] * y[:, None])

@triton.jit
def outer_matmul_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    # Define program IDs for x (row vector) and y (column vector)
    row_id = tl.program_id(0)  # ID for the row block
    col_id = tl.program_id(1)  # ID for the column block
    
    # Allocate shared memory for x and y vectors
    x_block = tl.load(x_ptr + row_id * B0 + tl.arange(0, B0))  # load B0 elements from x
    y_block = tl.load(y_ptr + col_id * B1 + tl.arange(0, B1))  # load B1 elements from y
    
    # Compute the output with ReLU
    for j in range(B1):  # iterate over elements in y
        result = tl.dot(x_block, y_block[j])  # outer product
        result = tl.maximum(result, 0)  # apply ReLU
        z_ptr[row_id * N1 + col_id * B1 + j] = result  # store result

    
if __name__ == "__main__":
    _test_puzzle(outer_matmul_kernel,outer_matmul_spec, nelem={"N0": 100, "N1": 90})
