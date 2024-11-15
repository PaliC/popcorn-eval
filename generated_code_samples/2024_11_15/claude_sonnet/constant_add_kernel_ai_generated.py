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
    # compute pid for data parallel tasks
    pid = tl.program_id(axis=0)
    # number of elements to process per block
    block_start = pid * B0
    # offsets for this program instance
    offsets = block_start + tl.arange(0, B0)
    # create mask for bounds check
    mask = offsets < N0
    # load data
    x = tl.load(x_ptr + offsets, mask=mask)
    # add constant value of 10 
    output = x + 10
    # write back result
    tl.store(z_ptr + offsets, output, mask=mask)

    
if __name__ == "__main__":
    _test_puzzle(constant_add_kernel, constant_add, nelem={"N0": 32})
