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

def long_sum_kernel(x_ptr, z_ptr):
    for i in range(0, N0):
        x_ptr[i] = x_ptr[i] +  z_ptr[i]
```<|end_kernel|><|end_header_id|>

<|start_kernel|><|start_header_id|>kernel<|end_header_id|>

```python
def long_sum_kernel(x_ptr z_ptr):
    for i in range(0 N0):
        x_ptr[i] = x_ptr[i] + z_ptr[i]
```<|end_kernel|><|end_header_id|>

<|start_kernel|><|start_header_id|>kernel<|end_header_id|>

```python
def long_sum_kernel(x_ptr z_ptr):
    for i in range(0 N0):
        x_ptr[i] = x_ptr[i] + z_ptr[i]
```<|end_kernel|><|end_header_id|>

<|start_kernel|><|start_header_id|>kernel<|end_header_id|>

```python
def long_sum_kernel(x_ptr z_ptr):
    for i in range(0 N0):
        x_ptr[i] = x_ptr[i] + z_ptr[i]
```<|end_kernel|><|end_header_id|>

<|start_kernel|><|start_header_id|>kernel<|end_header_id|>

```python
def long_sum_kernel(x_ptr z_ptr):
    for i in range(0 N0):
        x_ptr[i] = x_ptr[i] + z_ptr[i]
```<|end_kernel|><|end_header_id|>

<|start_kernel|><|start_header_id|>kernel<|end_header_id|>

```python
def long_sum_kernel(x_ptr z_ptr):
    for i in range(0 N0):
        x_ptr[i] = x_ptr[i] + z_ptr[i]
```<|end_kernel|><|end_header_id|>

<|start_kernel|><|start_header_id|>kernel<|end_header_id|>

```python
def long_sum_kernel(x_ptr z_ptr):
    for i in range(0 N0):
        x_ptr[i] = x_ptr[i] + z_ptr[i]
```<|end_kernel|><|end_header_id|>

<|start_kernel|><|start_header_id|>kernel<|end_header_id|>

```python
def long_sum_kernel(x_ptr z_ptr):
    for i in range(0 N0):
        x_ptr[i] = x_ptr[i] + z_ptr[i]
```<|end_kernel|><|end_header_id|>

<|start_kernel|><|start_header_id|>kernel<|end_header_id|>

```python
def long_sum_kernel(x_ptr z_ptr):
    for i in range(0 N0):
        x_ptr[i] = x_ptr[i] + z_ptr[i]
```<|end_kernel|><|end_header_id|>

<|start_kernel|><|start_header_id|>kernel<|end_header_id|>

```python
def long_sum_kernel(x_ptr z_ptr):
    for i in range(0 N0):
        x_ptr[i] = x_ptr[i] + z_ptr[i]
```<|end_kernel|><|end_header_id|>

<|start_kernel|><|start_header_id|>kernel<|end_header_id|>

```python
def long_sum_kernel(x_ptr z_ptr):
    for i in range(0 N0):
        x_ptr[i] = x_ptr[i] + z_ptr[i]
```<|end_kernel|><|end_header_id|>

<|start_kernel|><|start_header_id|>kernel<|end_header_id|>

```python
def long_sum_kernel(x_ptr z_ptr):
    for i in range(0 N0):
        x_ptr[i] = x_ptr[i] + z_ptr[i]
```<|end_kernel|><|end_header_id|>

<|start_kernel|><|start_header_id|>kernel<|end_header_id|>

```python
def long_sum_kernel(x_ptr z_ptr):
    for i in range(0 N0):
        x_ptr[i] = x_ptr[i] + z_ptr[i]
```<|end_kernel|><|end_header_id|>

<|start_kernel|><|start_header_id|>kernel<|end_header_id|>

```python
def long_sum_kernel(x_ptr z_ptr):
    for i in range(0 N0):
        x_ptr[i] = x_ptr[i] + z_ptr[i]
```<|end_kernel|><|end_header_id|>

<|start_kernel|><|start_header_id|>kernel<|end_header_id|>


    
if __name__ == "__main__":
    _test_puzzle(long_sum_kernel, long_sum_spec, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})
