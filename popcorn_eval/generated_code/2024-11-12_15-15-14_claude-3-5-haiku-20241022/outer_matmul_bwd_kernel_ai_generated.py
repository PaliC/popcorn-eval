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
    """
    Backward pass for outer matrix multiplication
    Compute gradients for x and y based on chain rule dz
    """
    # Pid for x and y vectors 
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    # Starting index offsets for x and y
    x_offset = pid0 * B0
    y_offset = pid1 * B1

    # Create block masks to handle boundary conditions
    x_mask = x_offset + tl.arange(0, B0) < N0
    y_mask = y_offset + tl.arange(0, B1) < N1

    # Load chain rule gradient dz with boundary checking
    dz_block = tl.load(z_ptr + x_offset + y_offset * N0, mask=x_mask & y_mask)

    # Compute x gradient 
    x_grad = tl.sum(dz_block, axis=1)
    tl.store(x_ptr + x_offset, x_grad, mask=x_mask)

    # Compute y gradient
    y_grad = tl.sum(dz_block, axis=0)
    tl.store(y_ptr + y_offset, y_grad, mask=y_mask)

    
if __name__ == "__main__":
    _test_puzzle(outer_matmul_bwd_kernel,outer_matmul_bwd_spec, nelem={"N0": 32})
