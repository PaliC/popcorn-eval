import triton
import triton.language as tl
import torch
import inspect
from _helper_functions import _puzzle_test
from jaxtyping import Float32, Tensor


# code taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles

def matmul_simple(x: Float32[Tensor, "4 32 32"], y: Float32[Tensor, "4 32 32"]) -> Float32[Tensor, "4 32 32"]:
    return x @ y

{{ GENERATED_CODE }}
    
if __name__ == "__main__":
    _puzzle_test(matmul_simple, matmul_simple_kernel, B={"B0": 16, "B1": 16, "B2": 1}, nelem={"N0": 32, "N1": 32, "N2": 4, "MID": 32})
