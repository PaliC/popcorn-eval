import triton
import triton.language as tl
import torch
import inspect
from _helper_functions import _puzzle_test
from jaxtyping import Float32, Tensor

# code taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles



def conv2d_spec(x: Float32[Tensor, "4 8 8"], k: Float32[Tensor, "4 4"]) -> Float32[Tensor, "4 8 8"]:
    z = torch.zeros(4, 8, 8)
    x = torch.nn.functional.pad(x, (0, 4, 0, 4, 0, 0), value=0.0)
    print(x.shape, k.shape)
    for i in range(8):
        for j in range(8):
            z[:, i, j] = (k[None, :, :] * x[:, i: i+4, j: j + 4]).sum(1).sum(1)
    return z


{{ GENERATED_CODE }}
    
if __name__ == "__main__":
   _puzzle_test( conv2d_spec,conv2d_kernel, B={"B0": 1}, nelem={"N0": 4, "H": 8, "W": 8, "KH": 4, "KW": 4})
