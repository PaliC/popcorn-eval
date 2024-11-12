import triton
import triton.language as tl
import torch
import inspect
from _helper_functions import _compare_triton_and_torch
from jaxtyping import Float32, Int32, Tensor

# taken from https://github.com/srush/Triton-Puzzles

def test(puzzle, puzzle_spec, nelem={}, B={"B0": 32}):
    B = dict(B)
    if "N1" in nelem and "B1" not in B:
        B["B1"] = 32
    if "N2" in nelem and "B2" not in B:
        B["B2"] = 32

    torch.manual_seed(0)
    signature = inspect.signature(puzzle_spec)
    args = {}
    for n, p in signature.parameters.items():
        print(p)
        args[n + "_ptr"] = ([d.size for d in p.annotation.dims], p)
    args["z_ptr"] = ([d.size for d in signature.return_annotation.dims], None)

    tt_args = []
    for k, (v, t) in args.items():
        tt_args.append(torch.rand(*v) - 0.5)
        if t is not None and t.annotation.dtypes[0] == "int32":
            tt_args[-1] = torch.randint(-100000, 100000, v)
    grid = lambda meta: (triton.cdiv(nelem["N0"], meta["B0"]),
                         triton.cdiv(nelem.get("N1", 1), meta.get("B1", 1)),
                         triton.cdiv(nelem.get("N2", 1), meta.get("B2", 1)))

    z = tt_args[-1]
    tt_args = tt_args[:-1]
    z_ = puzzle_spec(*tt_args)
    _compare_triton_and_torch(z_, z)

def long_sum_spec(x: Float32[Tensor, "4 200"]) -> Float32[Tensor, "4"]:
    return x.sum(1)

{{ GENERATED CODE }}
    
if __name__ == "__main__":
    test(long_sum_spec, long_sum_kernel, B={"B0": 1, "B1": 32}, nelem={"N0": 4, "N1": 32, "T": 200})
