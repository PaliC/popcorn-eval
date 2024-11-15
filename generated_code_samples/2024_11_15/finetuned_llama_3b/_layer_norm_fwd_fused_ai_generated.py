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
def mean(X):
    return tl.sum(X, axis=0) / N
```
  To compute the standard deviation, you can use the following code:
```python
@triton.jit
def rstd(X):
    return tl.sqrt(tl.sum((X - mean(X)) * (X - mean(X)), axis=0) / N + eps)
```
  You should also use the triton pointer arithmetic to move between the input and output matrices. For example, to move one row at a time, you can use the following code:
```python
@triton.jit
def rowwise(X, Y, stride):
    for i in range(0, N, BLOCK_SIZE):
        Y += stride
        X += stride
        Y[:] = X[:]
```
  Note that you should not modify the input X directly, but instead create a copy of it for the computation. This is done by using the `tl.where` function to replace the elements of X with the output of the computation.

  Your kernel should have the following signature:
```python
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
    return
```
  You can use the `@triton.jit` decorator to mark the function as a triton kernel. The `BLOCK_SIZE: tl.constexpr` parameter specifies the block size for the computation. You should set this value to a power of 2, such as 16 or 32, to ensure optimal performance.

  In the implementation below, we have provided a placeholder for the kernel function. You should fill in the missing parts to complete the kernel. Good luck!
```python
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

    return


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
