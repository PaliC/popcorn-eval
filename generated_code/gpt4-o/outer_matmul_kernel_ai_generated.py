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

def outer_matmul_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    # Get program ids
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    # Define block ranges
    offs_x = pid0 * B0 + tl.arange(0, B0)
    offs_y = pid1 * B1 + tl.arange(0, B1)

    # Create masks for boundary checking
    mask_x = offs_x < N0
    mask_y = offs_y < N1

    # Load elements from x and y vectors
    # x is a row vector, repeat each element B1 times
    x = tl.load(x_ptr + offs_x, mask=mask_x, other=0.0)
    x = x[:, None]  # Change shape to (B0, 1)
    # y is a column vector, repeat each element B0 times
    y = tl.load(y_ptr + offs_y, mask=mask_y, other=0.0)
    y = y[None, :]  # Change shape to (1, B1)

    # Multiply and take ReLU
    z = tl.relu(x * y)

    # Store the result
    offs_z = (pid0 * B0)[:, None] * N1 + (pid1 * B1)[None, :] + tl.arange((B0, B1))
    tl.store(z_ptr + offs_z, z, mask=mask_x[:, None] & mask_y[None, :])

    
if __name__ == "__main__":
    _test_puzzle(outer_matmul_kernel,outer_matmul_spec, nelem={"N0": 100, "N1": 90})
