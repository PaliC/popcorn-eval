import inspect
import sys

import torch
import torch.nn.functional as F
import triton


def _compare_triton_and_torch(triton_output, torch_output):
    rtol = 0
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
    # get cosine similarity
    cosine_similarity = F.cosine_similarity(triton_output, torch_output, dim=0)
    # print to standard error
    print(f"Cosine similarity: {cosine_similarity}")


def _test_puzzle(puzzle, puzzle_spec, nelem={}, B={"B0": 32}):
    B = dict(B)
    if "N1" in nelem and "B1" not in B:
        B["B1"] = 32
    if "N2" in nelem and "B2" not in B:
        B["B2"] = 32

    torch.manual_seed(0)
    signature = inspect.signature(puzzle_spec)
    args = {}
    for n, p in signature.parameters.items():
        args[n + "_ptr"] = ([d.size for d in p.annotation.dims], p)
    args["z_ptr"] = ([d.size for d in signature.return_annotation.dims], None)

    tt_args = []
    for k, (v, t) in args.items():
        tt_args.append(torch.rand(*v).to("cuda") - 0.5)
        if t is not None and t.annotation.dtypes[0] == "int32":
            tt_args[-1] = torch.randint(-100000, 100000, v)
    grid = lambda meta: (
        triton.cdiv(nelem["N0"], meta["B0"]),
        triton.cdiv(nelem.get("N1", 1), meta.get("B1", 1)),
        triton.cdiv(nelem.get("N2", 1), meta.get("B2", 1)),
    )

    # puzzle[grid](*tt_args, **B, **nelem)
    puzzle[grid](*tt_args, **B, **nelem)
    z = tt_args[-1]
    tt_args = tt_args[:-1]
    z_ = puzzle_spec(*tt_args)
    _compare_triton_and_torch(z_, z)
