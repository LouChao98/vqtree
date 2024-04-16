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
    q1,
    q2,
    sm_scale,
    K1_block_ptr,
    K2_block_ptr,
    V_block_ptr,
    start_m,
    offs_m,
    offs_n,
    SEQLEN_K: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_MN: tl.constexpr,
    STAGE: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        hi = start_m * BLOCK_M - WINDOW_SIZE + BLOCK_M
        lo = start_m * BLOCK_M - WINDOW_SIZE
        if hi < 0:
            hi = 0
        if lo < 0:
            lo = 0
    elif STAGE == 2:
        hi = start_m * BLOCK_M
        lo = start_m * BLOCK_M - WINDOW_SIZE + BLOCK_M
        if lo < 0:
            lo = 0
    else:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        hi = min(hi, SEQLEN_K)
    EVEN_MASK_FREE = EVEN_MN & ((STAGE == 1) | (STAGE == 2))

    K1_block_ptr = tl.advance(K1_block_ptr, (0, lo))
    K2_block_ptr = tl.advance(K2_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_MASK_FREE:
            k1 = tl.load(K1_block_ptr)
        else:
            k1 = tl.load(K1_block_ptr, boundary_check=(1,), padding_option="zero")
        qk = tl.dot(q1, k1) * (sm_scale * RCP_LN2)
        if STAGE == 1:
            if EVEN_MN:
                k2 = tl.load(K2_block_ptr)
            else:
                k2 = tl.load(K2_block_ptr, boundary_check=(1,), padding_option="zero")
            qk2 = tl.dot(q2, k2) * (sm_scale * RCP_LN2)
            mask = offs_m[:, None] <= start_n + WINDOW_SIZE + offs_n[None, :]
            qk = tl.where(mask, qk, qk2)
        elif STAGE == 3:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk += tl.where(mask, 0, NEGINF)
        if not EVEN_MASK_FREE:
            qk += tl.where((start_n + offs_n)[None, :] < SEQLEN_K, 0, NEGINF)
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
        K1_block_ptr = tl.advance(K1_block_ptr, (0, BLOCK_N))
        K2_block_ptr = tl.advance(K2_block_ptr, (0, BLOCK_N))
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
    # fmt:off
    Q1, Q2, K1, K2, V,
    Out, L, softmax_scale,
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
    WINDOW_SIZE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WRITE_LSE: tl.constexpr,
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
    Q1_block_ptr = tl.make_block_ptr(
        base=Q1 + (off_b * stride_qb + off_h * stride_qh),
        shape=(seqlen_q, BLOCK_HEADDIM),
        strides=(stride_qm, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_HEADDIM),
        order=(1, 0),
    )
    Q2_block_ptr = tl.make_block_ptr(
        base=Q2 + (off_b * stride_qb + off_h * stride_qh),
        shape=(seqlen_q, BLOCK_HEADDIM),
        strides=(stride_qm, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_HEADDIM),
        order=(1, 0),
    )
    K1_block_ptr = tl.make_block_ptr(
        base=K1 + (off_b * stride_kb + off_h * stride_kh),
        shape=(BLOCK_HEADDIM, seqlen_k),
        strides=(1, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_HEADDIM, BLOCK_N),
        order=(0, 1),
    )
    K2_block_ptr = tl.make_block_ptr(
        base=K2 + (off_b * stride_kb + off_h * stride_kh),
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
        q1 = tl.load(Q1_block_ptr)
        q2 = tl.load(Q2_block_ptr)
    else:
        q1 = tl.load(Q1_block_ptr, boundary_check=(0,), padding_option="zero")
        q2 = tl.load(Q2_block_ptr, boundary_check=(0,), padding_option="zero")

    # fmt: off
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q1, q2, softmax_scale, K1_block_ptr, K2_block_ptr, V_block_ptr, start_m, 
        offs_m, offs_n, seqlen_k, WINDOW_SIZE, BLOCK_M, BLOCK_N, EVEN_M & EVEN_N, 1)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q1, q2, softmax_scale, K1_block_ptr, K2_block_ptr, V_block_ptr, start_m, 
        offs_m, offs_n, seqlen_k, WINDOW_SIZE, BLOCK_M, BLOCK_N, EVEN_M & EVEN_N, 2)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q1, q2, softmax_scale, K1_block_ptr, K2_block_ptr, V_block_ptr, start_m, 
        offs_m, offs_n, seqlen_k, WINDOW_SIZE, BLOCK_M, BLOCK_N, EVEN_M & EVEN_N, 3)
    # fmt: on

    if WRITE_LSE:
        l_ptrs = L + off_hb * seqlen_q + offs_m
        tl.store(l_ptrs, m_i + tl.math.log2(l_i))

    acc = acc / l_i[:, None]
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    if EVEN_M:
        tl.store(out_ptrs, acc)
    else:
        tl.store(out_ptrs, acc, mask=offs_m[:, None] < seqlen_q)


def _flash_attn_forward(q1, q2, k1, k2, v, window_size, softmax_scale=None, return_lse=False):
    # shape constraints
    batch, seqlen_q, nheads, d = q1.shape
    _, seqlen_k, _, _ = k1.shape

    assert k1.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q1.dtype == q2.dtype == k1.dtype == k2.dtype == v.dtype, "All tensors must have the same type"
    assert q1.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q1.is_cuda and q2.is_cuda and k1.is_cuda and k2.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    o = torch.empty_like(q1, memory_format=torch.contiguous_format)
    if return_lse:
        lse = torch.empty((batch, nheads, seqlen_q), device=q1.device, dtype=torch.float32)
    else:
        lse = torch.empty((1,), device=o.device)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    assert BLOCK_HEADDIM == d

    device = torch.cuda.device_of(q1)

    with torch.cuda.device(device):

        BLOCK = 128
        num_warps = 4 if d <= 64 else 8
        grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
        _fwd_kernel[grid](
            # fmt: off
            q1, q2, k1, k2, v,
            o, lse, softmax_scale,
            q1.stride(0), q1.stride(2), q1.stride(1),
            k1.stride(0), k1.stride(2), k1.stride(1),
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
            WRITE_LSE=return_lse,
            num_warps=num_warps,
            num_stages=1,
        )
        if return_lse:
            return o, lse
        return o


def flash_attn_func(q1, q2, k1, k2, v, window_size, softmax_scale=None, return_lse=False):
    """
    q: (batch_size, seqlen_q, nheads, headdim)
    k, v: (batch_size, seqlen_k, nheads, headdim)
    """
    # Make sure that the last dimension is contiguous
    q1, q2, k1, k2, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q1, q2, k1, k2, v]]
    return _flash_attn_forward(q1, q2, k1, k2, v, window_size, softmax_scale=softmax_scale, return_lse=return_lse)
