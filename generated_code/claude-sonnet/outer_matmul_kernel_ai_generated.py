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
    """Computes outer product z = relu(x @ y) where x is row vector and y is column vector"""
    
    # program id and number of blocks
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    
    # offset pointers by the block index
    start_x = pid0 * B0 
    start_y = pid1 * B1
    
    # create mask for bounds check 
    mask_x = tl.arange(0, B0) < (N0 - start_x)
    mask_y = tl.arange(0, B1) < (N1 - start_y)
    
    # load vectors with masking
    x = tl.load(x_ptr + start_x, mask=mask_x)
    y = tl.load(y_ptr + start_y, mask=mask_y)
    
    # compute product for each element in block
    for i in range(B0):
        for j in range(B1):
            if mask_x[i] & mask_y[j]:
                # apply relu after multiply
                res = tl.maximum(x[i] * y[j], 0.0)
                # store result
                tl.store(z_ptr + start_x + i + (start_y + j) * N0, res)

    
if __name__ == "__main__":
    _test_puzzle(outer_matmul_kernel,outer_matmul_spec, nelem={"N0": 100, "N1": 90})
