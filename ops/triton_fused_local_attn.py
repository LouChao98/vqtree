"""Fused attn fwd
Based on https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py
"""

import math

import torch
import triton
import triton.language as tl

NEGINF = float("-inf")
RCP_LN2 = 1.4426950408889634
LN2 = 0.6931471824645996


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    sm_scale,
    K_block_ptr,
    V_block_ptr,
    start_m,
    offs_m,
    offs_n,
    seqlen_k: tl.constexpr,
    window_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_MN: tl.constexpr,
    STAGE: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        hi = start_m * BLOCK_M - window_size + BLOCK_M
        lo = start_m * BLOCK_M - window_size
        if hi < 0:
            hi = 0
        if lo < 0:
            lo = 0
    elif STAGE == 2:
        hi = start_m * BLOCK_M
        lo = start_m * BLOCK_M - window_size + BLOCK_M
        if lo < 0:
            lo = 0
    else:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        hi = min(hi, seqlen_k)
    EVEN_MASK_FREE = EVEN_MN & ((STAGE == 1) | (STAGE == 2))

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_MASK_FREE:
            k = tl.load(K_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        qk = tl.dot(q, k) * (sm_scale * RCP_LN2)
        if STAGE == 1:
            mask = offs_m[:, None] <= start_n + window_size + offs_n[None, :]
            qk += tl.where(mask, 0, NEGINF)
        elif STAGE == 3:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk += tl.where(mask, 0, NEGINF)
        if not EVEN_MASK_FREE:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, NEGINF)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])

        # -- scale and update acc --
        acc *= alpha[:, None]
        if EVEN_MASK_FREE:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr, boundary_check=(1,), padding_option="zero")
        acc += tl.dot(p.to(V_block_ptr.dtype.element_ty), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    return acc, l_i, m_i


# Disabling autotune for now, set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
#         # This config has a race condition when EVEN_M == False, disabling it for now.
#         # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
#     ],
#     key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'IS_CAUSAL', 'BLOCK_HEADDIM']
# )
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    softmax_scale,
    # fmt: off
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    # fmt: on
    nheads,
    seqlen_q,
    seqlen_k,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    window_size: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # Initialize pointers to Q, K, V
    Q_block_ptr = tl.make_block_ptr(
        base=Q + (off_b * stride_qb + off_h * stride_qh),
        shape=(seqlen_q, BLOCK_HEADDIM),
        strides=(stride_qm, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_HEADDIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + (off_b * stride_kb + off_h * stride_kh),
        shape=(BLOCK_HEADDIM, seqlen_k),
        strides=(1, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_HEADDIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + (off_b * stride_vb + off_h * stride_vh),
        shape=(seqlen_k, BLOCK_HEADDIM),
        strides=(stride_vn, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_HEADDIM),
        order=(1, 0),
    )

    # initialize pointer to m and l
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + NEGINF
    acc = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    if EVEN_M:
        q = tl.load(Q_block_ptr)
    else:
        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    # fmt: off
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, softmax_scale, K_block_ptr, V_block_ptr, start_m, 
        offs_m, offs_n, seqlen_k, window_size, BLOCK_M, BLOCK_N, EVEN_M & EVEN_N, 1)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, softmax_scale, K_block_ptr, V_block_ptr, start_m, 
        offs_m, offs_n, seqlen_k, window_size, BLOCK_M, BLOCK_N, EVEN_M & EVEN_N, 2)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, softmax_scale, K_block_ptr, V_block_ptr, start_m, 
        offs_m, offs_n, seqlen_k, window_size, BLOCK_M, BLOCK_N, EVEN_M & EVEN_N, 3)
    # fmt: on

    acc = acc / l_i[:, None]
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    if EVEN_M:
        tl.store(out_ptrs, acc)
    else:
        tl.store(out_ptrs, acc, mask=offs_m[:, None] < seqlen_q)


def _flash_attn_forward(q, k, v, window_size, softmax_scale=None):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape

    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    assert BLOCK_HEADDIM == d

    device = torch.cuda.device_of(q)

    with torch.cuda.device(device):

        BLOCK = 128
        num_warps = 4 if d <= 64 else 8
        grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
        _fwd_kernel[grid](
            q,
            k,
            v,
            o,
            softmax_scale,
            # fmt: off
            q.stride(0), q.stride(2), q.stride(1),
            k.stride(0), k.stride(2), k.stride(1),
            v.stride(0), v.stride(2), v.stride(1),
            o.stride(0), o.stride(2), o.stride(1),
            # fmt: on
            nheads,
            seqlen_q,
            seqlen_k,
            seqlen_q // 32,
            seqlen_k // 32,  # key for triton cache (limit number of compilations)
            # Can't use kwargs here because triton autotune expects key to be args, not kwargs
            # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
            window_size,
            BLOCK_HEADDIM,
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )
        return o


def flash_attn_func(q, k, v, window_size, softmax_scale=None):
    """
    q: (batch_size, seqlen_q, nheads, headdim)
    k, v: (batch_size, seqlen_k, nheads, headdim)
    """
    # Make sure that the last dimension is contiguous
    q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
    o = _flash_attn_forward(q, k, v, window_size, softmax_scale=softmax_scale)
    return o


if __name__ == "__main__":
    from flash_attn import flash_attn_func as ref_impl

    Z = 1
    H = 2
    D_HEAD = 64
    WINDOW = 512
    sm_scale = 0.5

    print("even")
    N_CTX = 2048
    q = torch.empty((Z, N_CTX, H, D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
    k = torch.empty((Z, N_CTX, H, D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
    v = torch.empty((Z, N_CTX, H, D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)

    ref_output = ref_impl(q, k, v, window_size=(WINDOW, 0), causal=True, softmax_scale=sm_scale)
    our_output = flash_attn_func(q, k, v, WINDOW, sm_scale)
    print((ref_output - our_output).norm())
    assert torch.allclose(ref_output, our_output, atol=2e-2, rtol=0)
    
    print("uneven")
    N_CTX = 2013
    q = torch.empty((Z, N_CTX, H, D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
    k = torch.empty((Z, N_CTX, H, D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)
    v = torch.empty((Z, N_CTX, H, D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5)

    ref_output = ref_impl(q, k, v, window_size=(WINDOW, 0), causal=True, softmax_scale=sm_scale)
    our_output = flash_attn_func(q, k, v, WINDOW, sm_scale)
    print((ref_output - our_output).norm())
    assert torch.allclose(ref_output, our_output, atol=2e-2, rtol=0)
