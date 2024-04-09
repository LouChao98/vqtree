import torch
from ops.triton_fused_attn import flash_attn_func
# from fused_attn import flash_attn_func as flash_attn_func2
from ops.triton_fused_attn2 import flash_attn_func as flash_attn_func2

q = torch.randn(17, 512, 12, 64).cuda().to(torch.float16)
k = torch.randn(17, 512, 12, 64).cuda().to(torch.float16)
v = torch.randn(17, 512, 12, 64).cuda().to(torch.float16)


o = flash_attn_func(q, k, v)
o2 = flash_attn_func2(q, k, v)

print("EVEN_M, EVEN_N", torch.allclose(o, o2))


q = torch.randn(17, 512, 12, 64).cuda().to(torch.float16)
k = torch.randn(17, 510, 12, 64).cuda().to(torch.float16)
v = torch.randn(17, 510, 12, 64).cuda().to(torch.float16)


o = flash_attn_func(q, k, v)
o2 = flash_attn_func2(q, k, v)

print("EVEN_M", torch.allclose(o, o2))


q = torch.randn(17, 490, 12, 64).cuda().to(torch.float16)
k = torch.randn(17, 510, 12, 64).cuda().to(torch.float16)
v = torch.randn(17, 510, 12, 64).cuda().to(torch.float16)


o = flash_attn_func(q, k, v)
o2 = flash_attn_func2(q, k, v)

print("", torch.allclose(o, o2))
