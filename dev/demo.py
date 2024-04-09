import torch
import triton
import triton.language as tl


B = 1
H = 1
D = 64 * H
L = 512
C = 65536  # This is the memory size.
S = 256  # S^M = L
M = 2

query = torch.randn(B, H, L, D)

k_cache = torch.randn(B, H, sum(S**i for i in range(1, M + 1)), D)
k_cnt = torch.randint(B, H, sum(S**i for i in range(1, M + 1)), dtype=torch.int32)
v_cache = torch.randn(B, H, sum(S**i for i in range(1, M + 1)), D)
v_cnt = torch.randint(B, H, sum(S**i for i in range(1, M + 1)), dtype=torch.int32)


