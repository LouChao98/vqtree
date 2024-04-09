"""Fused attn fwd on AD tree
Based on https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py
"""

import math

import torch
import triton
import triton.language as tl

NEGINF = float("-inf")


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
        "TOTAL_SLOTS": lambda args: sum(
            args["CODEBOOK_SIZE"] ** i for i in range(1, args["CODEBOOK_NUM"] + 1)
        ),
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    CODEBOOK_K,  # B x H x (S x M) x D
    CODEBOOK_V,
    KCNT,
    VCNT,
    Out,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    CODEBOOK_SIZE: tl.constexpr,
    CODEBOOK_NUM: tl.constexpr,
    TOTAL_SLOTS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
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
        shape=(TOTAL_SLOTS, BLOCK_HEADDIM),
        strides=(stride_kn, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_HEADDIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + (off_b * stride_vb + off_h * stride_vh),
        shape=(TOTAL_SLOTS, BLOCK_HEADDIM),
        strides=(stride_vn, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_HEADDIM),
        order=(1, 0),
    )

    # initialize pointer to m and l
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) + NEGINF
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + NEGINF
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    
    # load q: it will stay in SRAM throughout
    if EVEN_M:
        q = tl.load(Q_block_ptr)
    else:
        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    
    offset = 0
    
    for level_idx in range(1, CODEBOOK_NUM + 1):
        
        K_inner_block_ptr = K_block_ptr
        V_inner_block_ptr = V_block_ptr
        
        # record shortcode
        z_value = tl.zeros([BLOCK_M], dtype=tl.float32) + NEGINF
        z_idx = tl.zeros([BLOCK_M], dtype=tl.int32)
    
        # attn to a codebook
        for start_n in range(0, CODEBOOK_SIZE, BLOCK_N):
            # -- compute qk ----
            k = tl.load(K_inner_block_ptr)
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            
            qk += tl.dot(q, tl.trans(k))
            qk_max = tl.max(qk, 1)
            m_ij = tl.maximum(qk_max * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            
            # -- update codebook --
            qk_max_indices = tl.argmax(qk, 1) + start_n
            update_codebook = z_value < qk_max
            z_idx = tl.where(update_codebook, qk_max_indices, z_idx)
            z_value = tl.where()
            
            # -- scale acc_o --
            acc_o_scale = tl.exp(m_i - m_ij)
            
            # -- update output accumulator --
            acc_o = acc_o * acc_o_scale[:, None]
            
            # -- update acc_o --
            v = tl.load(V_inner_block_ptr)
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)
            
            # -- update statistics --
            m_i = m_ij
            l_i_new = tl.exp(lse_i - m_ij) + l_ij
            lse_i = m_ij + tl.log(l_i_new)
            
            # -- advance ptrs --            
            K_inner_block_ptr = tl.advance(K_inner_block_ptr, (BLOCK_N, 0))
            V_inner_block_ptr = tl.advance(V_inner_block_ptr, (BLOCK_N, 0))
        
        
    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        # If we just do "if EVEN_N", there seems to be some race condition
        if EVEN_N & EVEN_M:
            k = tl.load(K_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, NEGINF)
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, NEGINF)
        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator --
        acc_o = acc_o * acc_o_scale[:, None]
        # update acc_o
        # If we just do "if EVEN_N", there seems to be some race condition
        if EVEN_N & EVEN_M:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back m
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        tl.store(out_ptrs, acc_o)
    else:
        tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)


def _flash_attn_forward(q, k, v, causal=False, softmax_scale=None):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    print(seqlen_q, seqlen_k)
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
            q.stride(0),
            q.stride(2),
            q.stride(1),
            k.stride(0),
            k.stride(2),
            k.stride(1),
            v.stride(0),
            v.stride(2),
            v.stride(1),
            o.stride(0),
            o.stride(2),
            o.stride(1),
            nheads,
            seqlen_q,
            seqlen_k,
            seqlen_q // 32,
            seqlen_k // 32,  # key for triton cache (limit number of compilations)
            # Can't use kwargs here because triton autotune expects key to be args, not kwargs
            # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
            causal,
            BLOCK_HEADDIM,
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )
        return o


def flash_attn_func(q, k, v, causal=False, softmax_scale=None):
    """
    q: (batch_size, seqlen_q, nheads, headdim)
    k, v: (batch_size, seqlen_k, nheads, headdim)
    """
    # Make sure that the last dimension is contiguous
    q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
    o = _flash_attn_forward(q, k, v, causal=causal, softmax_scale=softmax_scale)
    return o
