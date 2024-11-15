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
    # compute pid and block start indices
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    
    # offset block start
    offs0 = pid0 * B0 + tl.arange(0, B0)
    offs1 = pid1 * B1 + tl.arange(0, B1)
    
    # create mask for bounds check
    mask0 = offs0 < N0
    mask1 = offs1 < N1
    
    # load vectors with mask
    x = tl.load(x_ptr + offs0, mask=mask0)
    y = tl.load(y_ptr + offs1, mask=mask1)
    
    # compute outer product for this block
    z = tl.zeros([B0, B1], dtype=tl.float32)
    for i in range(B0):
        for j in range(B1):
            # mask for valid indices
            mask = mask0[i] & mask1[j]  
            # multiply elements
            prod = x[i] * y[j]
            # store with relu
            z[i,j] = tl.where(prod > 0, prod, 0)
    
    # store output with mask
    offs_out = pid0 * B0 * N1 + pid1 * B1 
    for i in range(B0):
        for j in range(B1):
            mask = mask0[i] & mask1[j]
            tl.store(z_ptr + offs_out + i * N1 + j, z[i,j], mask=mask)

    
if __name__ == "__main__":
    _test_puzzle(outer_matmul_kernel,outer_matmul_spec, nelem={"N0": 100, "N1": 90})
