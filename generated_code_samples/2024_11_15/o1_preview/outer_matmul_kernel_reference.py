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
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    range0 = tl.arange(0, B0) + pid_0 * B0
    range1 = tl.arange(0, B1) + pid_1 * B1

    x = tl.load(x_ptr + range0, mask=range0 < N0)
    y = tl.load(y_ptr + range1, mask=range1 < N1)

    z = tl.maximum(x[None, :] * y[:, None], 0)

    range_block = range1[:, None] * N0 + range0[None, :]
    mask = (range0[None, :] < N0) & (range1[:, None] < N1)
    tl.store(z_ptr + range_block, z, mask=mask)

    return


    
if __name__ == "__main__":
    _test_puzzle(outer_matmul_kernel,outer_matmul_spec, nelem={"N0": 100, "N1": 90})
