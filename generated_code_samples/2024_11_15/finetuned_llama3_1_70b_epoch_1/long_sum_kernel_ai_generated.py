import triton
import triton.language as tl
import torch
import inspect
from _helper_functions import _test_puzzle
from jaxtyping import Float32, Int32
from torch import Tensor

# taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles

def long_sum_spec(x: Float32[Tensor, "4 200"]) -> Float32[Tensor, "4"]:
    return x.sum(1)

@triton.jit
@triton.heuristics.pointwise(size_hints=[B0], filename=__file__,
 triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32',
 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32',
'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32',
'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32',
'out_ptr1': '*fp32', 'xnumel': 'i32', 'rblock': 'i32'}, 'device':
 DeviceProperties(type='cuda', index=0, The kernel should have the function signature:
```python
def long_sum_kernel(x_ptr, z_ptr
    N0: tl.constexpr
    T: tl.constexpr
    B0: tl.constexpr
    B1: tl.constexpr):
```

```python
@triton.jit
@triton.heuristics.pointwise(size_hints=[B0], filename=__file__,
 triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32',
'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32',
'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32',
'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32',
'out_ptr1': '*fp32', 'xnumel': 'i32', 'rblock': 'i32'}, 'device':
 DeviceProperties(type='cuda', index=0)
```python
@triton.jit
@triton.heuristics.pointwise(size_hints=[B0], filename=__file__, triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32',
'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32',
'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32',
'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32',
'out_ptr1': '*fp32', 'xnumel': 'i32', 'rblock': 'i32'}, 'device':
 DeviceProperties(type='cuda', index=0)
```python
@triton.jit
@triton.heuristics.pointwise(size_hints=[B0], filename=__file__, triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32',
'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32',
'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32',
'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32',
'out_ptr1': '*fp32', 'xnumel': 'i32', 'rblock': 'i32'}, 'device':
 DeviceProperties(type='cuda', index=0)

    
if __name__ == "__main__":
    _test_puzzle(long_sum_kernel, long_sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})
