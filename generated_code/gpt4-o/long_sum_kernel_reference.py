import triton
import triton.language as tl
import torch
import inspect
from _helper_functions import _test_puzzle
from jaxtyping import Float32, Int32
from torch import Tensor

# taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles

def long_sum_spec(x: Float32[Tensor, "4 200"]) -> Float32[Tensor, "4"]:
    return x.sum(1)

@triton.jit
def long_sum_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    # Get program ID for the row dimension
    pid_0 = tl.program_id(0)

    # Calculate row offset and check if it's valid
    row_idx = pid_0 * B0 + tl.arange(0, B0)
    row_mask = row_idx < N0

    # Initialize accumulator for the sum
    accumulator = tl.zeros([B0], dtype=tl.float32)

    # Loop over columns in blocks
    for col_offset in range(0, T, B1):
        # Calculate column indices and mask for this block
        col_idx = col_offset + tl.arange(0, B1)
        col_mask = col_idx < T

        # Calculate the offsets into the input array
        offsets = row_idx[:, None] * T + col_idx[None, :]

        # Load values from input, masking invalid locations
        block = tl.load(
            x_ptr + offsets, mask=row_mask[:, None] & col_mask[None, :], other=0.0
        )

        # Add to accumulator
        accumulator += tl.sum(block, axis=1)

    # Store final results
    tl.store(z_ptr + row_idx, accumulator, mask=row_mask)

    
if __name__ == "__main__":
    _test_puzzle(long_sum_kernel, long_sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})
