import triton
import triton.language as tl
import torch
import inspect
from _helper_functions import _test_puzzle
from jaxtyping import Float32, Int32
from torch import Tensor

# taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles

def outer_matmul_bwd_spec(x: Float32[Tensor, "90 100"], y: Float32[Tensor, "90"],
                             dz: Float32[Tensor, "90 100"]) -> Float32[Tensor, "90 100"]:
    x = x.clone()
    y = y.clone()
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = torch.relu(x * y[:, None])
    z.backward(dz)
    dx = x.grad
    return dx


@triton.jit
def outer_matmul_bwd_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    # z = relu(x * y)
    # dL/dx = dL/dz * dz/dx
    #       = dL/dx * drelu(x*y)/d(x*y) * d(x*y)/dx
    #       = dL/dx * (x * y > 0) * y

    range_0 = tl.arange(0, B0) + pid_0 * B0
    range_1 = tl.arange(0, B1) + pid_1 * B1
    range_block = range_1[:, None] * N0 + range_0[None, :]
    mask_block = (range_0[None, :] < N0) & (range_1[:, None] < N1)

    x = tl.load(x_ptr + range_block, mask=mask_block)
    y = tl.load(y_ptr + range_1, mask=range_1 < N1)
    dz = tl.load(dz_ptr + range_block, mask=mask_block)

    dx = dz * (x * y[:, None] > 0) * y[:, None]

    tl.store(dx_ptr + range_block, dx, mask=mask_block)

    return

    
if __name__ == "__main__":
    _test_puzzle(outer_matmul_bwd_kernel,outer_matmul_bwd_spec, nelem={"N0": 32})
