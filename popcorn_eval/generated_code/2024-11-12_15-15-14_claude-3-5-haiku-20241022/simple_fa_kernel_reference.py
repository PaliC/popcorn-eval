import triton
import triton.language as tl
import torch
import inspect
from _helper_functions import _test_puzzle
from jaxtyping import Float32, Int32
from torch import Tensor

# taken from https://github.com/srush/Triton-Puzzles and https://github.com/gau-nernst/Triton-Puzzles

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

def simple_fa_spec(q: Float32[Tensor, "200"], k: Float32[Tensor, "200"], v: Float32[Tensor, "200"]) -> Float32[Tensor, "200"]:
    x = q[:, None] * k[None, :]
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    soft =  x_exp  / x_exp.sum(1, keepdim=True)
    return (v[None, :] * soft).sum(1)

@triton.jit
def flashatt_kernel(q_ptr, k_ptr, v_ptr, z_ptr, N0, T, B0: tl.constexpr):
    log2_e = 1.44269504

    # load q_block (B0,)
    # load k_block (B0,)
    # calculate outer product qk (B0, B0)
    # do max-sum

    for q_offset in tl.static_range(0, N0, B0):
        q_range = q_offset + tl.arange(0, B0)
        q_block = tl.load(q_ptr + q_range, mask=q_range < N0)

        _max = (tl.arange(0, B0) + 1) * (-1e10)
        _denom = tl.arange(0, B0) * 0
        z_block = tl.arange(0, B0) * 0

        for k_offset in tl.static_range(0, N0, B0):
            k_range = k_offset + tl.arange(0, B0)
            k_block = tl.load(k_ptr + k_range, mask=k_range < N0)

            qk_block = q_block[:, None] * k_block[None, :]

            # mask so that maximum is correct
            qk_block = qk_block * (k_range < N0) + (-1e10) * (k_range >= N0)

            new_max = tl.maximum(_max, tl.max(qk_block, axis=1))
            exp_correction = tl.exp2(_max - new_max)
            qk_block_exp = tl.exp2((qk_block - new_max[:, None]) * log2_e)

            new_denom = _denom * exp_correction + tl.sum(qk_block_exp, axis=1)
            qk_block_prob = qk_block_exp / new_denom[:, None]

            v_block = tl.load(v_ptr + k_range, mask=k_range < N0)
            z_block *= exp_correction * _denom / new_denom
            z_block += tl.sum(qk_block_prob * v_block[None, :], axis=1)

            _max = new_max
            _denom = new_denom

        tl.store(z_ptr + q_range, z_block, mask=q_range < N0)

    return

    
if __name__ == "__main__":
    _test_puzzle(simple_fa_kernel, simple_fa_spec, B={"B0": 1, "B1":32},
     nelem={"N0": 4, "N1": 32, "T": 200})
