import triton
import triton.language as tl
import torch
import inspect
from _helper_functions import _puzzle_test
from jaxtyping import Float32, Tensor


# code taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles

FPINT = 32 // 4
GROUP = 8

def quantized_matmul_simple(scale : Float32[Tensor, "32 8"],
                   offset : Int32[Tensor, "32"],
                   weight: Int32[Tensor, "32 8"],  # in fact (32, 64)
                   activation: Float32[Tensor, "64 32"]) -> Float32[Tensor, "32 32"]:
    offset = offset.view(32, 1)
    def extract(x):
        over = torch.arange(8) * 4
        mask = 2**4 - 1
        return (x[..., None] >> over) & mask
    scale = scale[..., None].expand(-1, 8, GROUP).contiguous().view(-1, 64)
    offset = extract(offset)[..., None].expand(-1, 1, 8, GROUP).contiguous().view(-1, 64)
    return ( scale * (extract(weight).view(-1, 64) - offset))  @ activation

{{ GENERATED_CODE }}
    
if __name__ == "__main__":
    _puzzle_test(quantized_matmul_simple, quantized_matmul_simple_kernel, B={"B0": 16, "B1": 16, "B_MID": 64}, nelem={"N0": 32, "N1": 32, "MID": 64})
