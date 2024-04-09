"""Fused attn fwd
Based on https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py
"""

import math
from sympy import N

import torch
import triton
import triton.language as tl

from flashinfer import merge_state_in_place
from ops.triton_fused_local_attn2 import flash_attn_func as block_local_attention

NEGINF = -1e6
# NEGINF = float("-inf")
RCP_LN2 = 1.4426950408889634
LN2 = 0.6931471824645996


@triton.jit
def _vq_attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    sm_scale,
    K_block_ptr,
    K_VQC_block_ptr,
    VVQI_block_ptr,
    V_VQ,
    stride_vvq_n,
    CODEBOOK_SIZE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # loop over k, v and update accumulator
    for _ in range(0, CODEBOOK_SIZE, BLOCK_N):

        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k) * (sm_scale * RCP_LN2)

        k_vq_cnt = tl.load(K_VQC_block_ptr)
        mask = k_vq_cnt != 0.
        qk = tl.where(mask[None, :], qk, NEGINF)

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])

        # -- scale and update acc --
        acc *= alpha[:, None]
        v_vq_index = tl.load(VVQI_block_ptr)
        v_ptr = V_VQ + (v_vq_index[:, None] * stride_vvq_n + tl.arange(0, BLOCK_HEADDIM)[None, :])
        v = tl.load(v_ptr)
        acc += tl.dot(p.to(v.dtype), v)
        # acc += tl.sum((mask * 1.).to(v.dtype)[:, None] * v, axis=0)

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p.to(v.dtype) * k_vq_cnt[None, :], axis=1)
        m_i = m_i_new

        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        K_VQC_block_ptr = tl.advance(K_VQC_block_ptr, (BLOCK_N,))
        VVQI_block_ptr = tl.advance(VVQI_block_ptr, (BLOCK_N,))

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
    }
)
@triton.jit
def _vq_fwd_kernel(
    # fmt: off
    Q, K_VQ, K_VQ_CNT, V_VQ, V_VQ_INDEX,
    Out, L, softmax_scale,
    stride_q_b,     stride_q_h,     stride_q_m,
                    stride_kvq_h,   stride_kvq_c,
    stride_kvqc_b,  stride_kvqc_h,  stride_kvqc_n,
    stride_vvq_b,   stride_vvq_h,   stride_vvq_n,
    stride_vvqi_b,  stride_vvqi_h,  stride_vvqi_n,
    stride_o_b,     stride_o_h,     stride_o_m,
    # fmt: on
    nheads,
    seqlen_q,
    codebook_size,
    CACHE_KEY_SEQLEN_Q,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + WINDOW_SIZE + BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # initialize pointer to m and l
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + NEGINF
    acc = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # ==========================================================================
    #  Attend to Vector Quantized KV cache
    # ==========================================================================

    Q_block_ptr = tl.make_block_ptr(
        base=Q + (off_b * stride_q_b + off_h * stride_q_h),
        shape=(seqlen_q, BLOCK_HEADDIM),
        strides=(stride_q_m, 1),
        offsets=(start_m * BLOCK_M + WINDOW_SIZE + BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_HEADDIM),
        order=(1, 0),
    )
    K_VQ_block_ptr = tl.make_block_ptr(
        base=K_VQ + off_h * stride_kvq_h,
        shape=(BLOCK_HEADDIM, codebook_size),
        strides=(1, stride_kvq_c),
        offsets=(0, 0),
        block_shape=(BLOCK_HEADDIM, BLOCK_N),
        order=(0, 1),
    )
    K_VQC_block_ptr = tl.make_block_ptr(
        base=K_VQ_CNT
        + (off_b * stride_kvqc_b + off_h * stride_kvqc_h + start_m * stride_kvqc_n),
        shape=(codebook_size,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )
    VVQI_block_ptr = tl.make_block_ptr(
        base=V_VQ_INDEX
        + (off_b * stride_vvqi_b + off_h * stride_vvqi_h + start_m * stride_vvqi_n),
        shape=(codebook_size,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )
    V_VQ += off_b * stride_vvq_b + off_h * stride_vvq_h
    if EVEN_M:
        q = tl.load(Q_block_ptr)
    else:
        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    # fmt: off
    acc, l_i, m_i = _vq_attn_fwd_inner(acc, l_i, m_i, q, softmax_scale, K_VQ_block_ptr, K_VQC_block_ptr,
        VVQI_block_ptr, V_VQ, stride_vvq_n, codebook_size, BLOCK_HEADDIM, BLOCK_N)
    # fmt: on

    acc = acc / l_i[:, None]
    
    # start_m = tl.program_id(0) + WINDOW_SIZE // BLOCK_M
    # offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # offs_d = tl.arange(0, BLOCK_HEADDIM)

    if WRITE_LSE:
        l_ptrs = L + off_hb * seqlen_q + offs_m
        tl.store(l_ptrs, m_i + tl.math.log2(l_i))

    out_ptrs = Out + off_b * stride_o_b + off_h * stride_o_h + (offs_m[:, None] * stride_o_m + offs_d[None, :])
    if EVEN_M:
        tl.store(out_ptrs, acc)
    else:
        tl.store(out_ptrs, acc, mask=offs_m[:, None] < seqlen_q)


def vq_attn_forward(q1, q2, k1, k_vq, k_vq_cnt, v, v_vq, vi, window_size, softmax_scale=None):
    # shape constraints

    q1, q2, k1, k_vq, k_vq_cnt, v, v_vq, vi = [
        x if x.stride(-1) == 1 else x.contiguous() for x in [q1, q2, k1, k_vq, k_vq_cnt, v, v_vq, vi]
    ]

    batch, nheads, seqlen_q, d = q1.shape
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    assert BLOCK_HEADDIM == d
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    o1, l1 = block_local_attention(
        q1.transpose(1, 2),  # TODO unify layout
        k1.transpose(1, 2),
        v.transpose(1, 2),
        window_size,
        softmax_scale,
        return_lse=True,
    )
    o1 = o1.transpose(1, 2)

    o2 = torch.zeros_like(q1, memory_format=torch.contiguous_format)
    l2 = torch.full((batch, nheads, seqlen_q), float('-inf'), device=q2.device, dtype=torch.float32)

    device = torch.cuda.device_of(q1)
    with torch.cuda.device(device):

        BLOCK = 128
        num_warps = 4 if d <= 64 else 8
        grid = lambda META: (triton.cdiv(seqlen_q - window_size - BLOCK, META["BLOCK_M"]), batch * nheads)
        _vq_fwd_kernel[grid](
            # fmt: off
            q2, k_vq, k_vq_cnt, v_vq, vi, 
            o2, l2, softmax_scale,
            # fmt: on
            *q2.stride()[:-1],
            *k_vq.stride()[:-1],
            *k_vq_cnt.stride()[:-1],
            *v_vq.stride()[:-1],
            *vi.stride()[:-1],
            *o2.stride()[:-1],
            nheads,
            seqlen_q,
            k_vq.shape[1],
            seqlen_q // 32,
            window_size,
            BLOCK_HEADDIM,
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            WRITE_LSE=True,
            num_warps=num_warps,
            num_stages=1,
        )
    o1 = o1.view(batch * nheads, seqlen_q, d).transpose(0, 1).contiguous()
    l1 = l1.view(batch * nheads, seqlen_q).transpose(0, 1).contiguous()
    o2 = o2.view(batch * nheads, seqlen_q, d).transpose(0, 1).contiguous()
    l2 = l2.view(batch * nheads, seqlen_q).transpose(0, 1).contiguous()
    merge_state_in_place(o1, l1, o2, l2)
    return o1.view(seqlen_q, batch, nheads, d).permute(1, 2, 0, 3)
