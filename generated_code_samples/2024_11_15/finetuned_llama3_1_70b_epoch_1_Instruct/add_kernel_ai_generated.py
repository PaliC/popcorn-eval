"""
Vector Addition
===============

In this tutorial, you will write a simple vector addition using Triton.

In doing so, you will learn about:

* The basic programming model of Triton.

* The `triton.jit` decorator, which is used to define Triton kernels.

* The best practices for validating and benchmarking your custom ops against native reference implementations.

"""

# %%
# Compute Kernel
# --------------

import torch

import triton
import triton.language as tl
from _helper_functions import _compare_triton_and_torch


@tl.program_id('add_kernel')
@tl.constexpr
def add_kernel(x_ptr, # *Pointer* to first input vector.
    y_ptr### *Pointer* to second input vector.
    output_ptr### *Pointer* to output vector.
    n_elements### Size of the vector.
    BLOCK_SIZE: tl.constexpr### Number of elements each program should process.
    ):
    x_ptr = x_ptr
    y_ptr = y_ptr
    output_ptr = output_ptr
    n_elements = n_elements
    BLOCK_SIZE = BLOCK_SIZE
    for i in range(0, # *Range* to *Size of the input vector* ):
        x = tl.load(x_ptr + i)
        y = tl.load(y_ptr + i)
        output = x + y
        tl.store(output_ptr + i, ## *Store* the result of the computation.
            output)


def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output


# %%
# We can now use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:

if __name__ == "__main__":

    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    output_torch = x + y
    output_triton = add(x, y)
    _compare_triton_and_torch(output_triton, output_torch)
