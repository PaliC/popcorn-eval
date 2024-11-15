# taken from https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html

from _helper_functions import _compare_triton_and_torch
import torch

import triton
import triton.language as tl

try:
    # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
    # should not be added to extras_require in setup.py.
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False
        
@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program id
    pid = tl.program_id(0)
    # Compute the row start index
    row_start = pid * BLOCK_SIZE
    # Create an index for the current row
    offs = row_start + tl.arange(0, BLOCK_SIZE)
    # Mask for valid columns
    mask = offs < N
    # Load the input elements
    x = tl.load(X + pid * stride + offs, mask=mask, other=0.0)
    # Initialize sum and sum of squares
    sum_x = tl.sum(x, mask=mask)
    sum_x2 = tl.sum(x * x, mask=mask)
    # Compute mean and variance
    mean = sum_x / N
    var = (sum_x2 / N) - (mean * mean)
    # Compute rstd
    rstd = tl.rsqrt(var + eps)
    # Store mean and rstd
    tl.store(Mean + pid, mean)
    tl.store(Rstd + pid, rstd)
    # Normalize, scale, and shift
    y = (x - mean) * rstd
    w = tl.load(W + offs, mask=mask, other=1.0)
    b = tl.load(B + offs, mask=mask, other=0.0)
    y = y * w + b
    # Store the output
    tl.store(Y + pid * stride + offs, y, mask=mask)


class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        _layer_norm_fwd_fused[(M, )](  #
            x_arg, y, weight, bias, mean, rstd,  #
            x_arg.stride(0), N, eps,  #
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

layer_norm = LayerNorm.apply

def test_layer_norm(M, N, dtype, eps=1e-5, device='cuda'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = layer_norm(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)

    _compare_triton_and_torch(y_tri, y_ref)

if __name__ == '__main__':
    test_layer_norm(1024, 1024, torch.float32, eps=1e-5, device='cuda')
