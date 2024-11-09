import sys

import torch
import torch.nn.functional as F


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
