import triton
import triton.language as tl
import torch
import inspect
from _helper_functions import _test_puzzle
from jaxtyping import Float32, Int32
from torch import Tensor

# taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles

def constant_add(x: Float32[Tensor, "32"]) -> Float32[Tensor, "32"]:
    return x + 10

@triton.jit
def constant_add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * B0
    offsets = block_start + tl.arange(0, B0)
    mask = offsets < N0
    x = tl.load(x_ptr + offsets, mask=mask)
    z = x + 10
    tl.store(z_ptr + offsets, z, mask=mask)

    
if __name__ == "__main__":
    _test_puzzle(constant_add_kernel, constant_add, nelem={"N0": 32})
