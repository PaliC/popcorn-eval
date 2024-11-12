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
def simple_fa_kernel(q_ptr, k_ptr, v_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    # Program IDs for block-level processing
    pid_n0 = tl.program_id(0)  # Block index for N0 dimension
    pid_n1 = tl.program_id(1)  # Block index for N1 dimension

    # Initialize base offsets for query, key, value, output
    offs_n0 = pid_n0 * B0 + tl.arange(0, B0)
    offs_n1 = pid_n1 * B1 + tl.arange(0, B1)

    # Load query and key blocks
    q = tl.load(q_ptr + offs_n0 * T + offs_n1, mask=offs_n1 < T)
    k = tl.load(k_ptr + offs_n0 * T + offs_n1, mask=offs_n1 < T)

    # Initialize output accumulator
    z = tl.zeros([B0, B1], dtype=tl.float32)

    # Single loop over sequence dimension
    for t in range(T):
        # Compute QK scores with broadcasting
        qk = q * k
        
        # Apply softmax (simplified for scalar version)
        exp_qk = tl.exp(qk)
        softmax_qk = exp_qk / tl.sum(exp_qk)

        # Load value and compute weighted contribution
        v = tl.load(v_ptr + offs_n0 * T + t, mask=offs_n0 < N0)
        z += softmax_qk * v

    # Store result
    tl.store(z_ptr + offs_n0 * N1 + offs_n1, z, mask=offs_n1 < N1)

    
if __name__ == "__main__":
    _test_puzzle(simple_fa_kernel, simple_fa_spec, B={"B0": 1, "B1":32},
     nelem={"N0": 4, "N1": 32, "T": 200})
