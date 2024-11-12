import triton
import triton.language as tl
import torch
import inspect
from _helper_functions import _test_puzzle

# taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles

def constant_add(x: Float32[Tensor, "32"]) -> Float32[Tensor, "32"]:
    return x + 10

@triton.jit
def constant_add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    range = tl.arange(0, B0)
    x = tl.load(x_ptr + range)
    x += 10
    tl.store(z_ptr + range, x)

    
if __name__ == "__main__":
    _test_puzzle(constant_add, constant_add_kernel, nelem={"N0": 32})
