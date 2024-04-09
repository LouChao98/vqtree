from enum import Enum
from functools import partial
import math
from typing import Optional, List, Dict, Any, Tuple

import torch
from transformers.generation.utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2,
    LLAMA_ATTENTION_CLASSES,
    apply_rotary_pos_emb,
    rotate_half,
    logger,
)

from flash_attn import flash_attn_func as naive_prefilling_kernel
from ops.triton_fused_vq_attn import vq_attn_forward as vq_prefilling_kernel
from contextlib import contextmanager

from ops.mini_vq import multi_head_single_quantize


class AttentionMode(Enum):
    SLOW_REFERNECE = 0
    SORT_BASED_LINEAR = 1


class VectorQuantizedReRoPECache(Cache):
    # A cache that as described in https://kexue.fm/archives/9708 + Transformer-VQ
    # Based on SinkCache

    def __init__(self, window_length: int, codebook, sphere=False, *, attention_mode=AttentionMode.SORT_BASED_LINEAR):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self.window_length = window_length
        self.codebook = codebook
        assert self.codebook.shape[1] % 128 == 0
        self.sphere = sphere
        self.cos_1 = {}
        self.sin_1 = {}
        self.seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

        # config for attention
        self.attention_mode = attention_mode

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_key_rotary_pos_emb(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        rotated_key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return rotated_key_states

    def _get_rerotation_cos_sin(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states.device not in self.cos_1:
            self.cos_1[key_states.device] = cos[1].to(key_states.device)
            self.sin_1[key_states.device] = sin[1].to(key_states.device)

        return (
            self.cos_1[key_states.device][None, None, :].expand(-1, key_states.shape[-2], -1),
            self.sin_1[key_states.device][None, None, :].expand(-1, key_states.shape[-2], -1),
        )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return int(1e8)  # some safe number

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        sin = cache_kwargs.get("sin")
        cos = cache_kwargs.get("cos")
        partial_rotation_size = cache_kwargs.get("partial_rotation_size")
        using_rope = cos is not None and sin is not None

        assert using_rope, "This is useless for models without RoPE."
        assert partial_rotation_size is None, "TODO"

        # Update the number of seen tokens
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]

        # [bsz, num_heads, seq_len, head_dim]
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        else:
            # Growing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

            if self.get_seq_length(layer_idx) > self.window_length:
                keys_to_update = self.key_cache[layer_idx][:, :, : -self.window_length]
                rerotation_cos, rerotation_sin = self._get_rerotation_cos_sin(keys_to_update, cos, sin)
                keys_to_update.copy_(self._apply_key_rotary_pos_emb(keys_to_update, rerotation_cos, rerotation_sin))

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))


class LlamaFlashAttention2VectorQuantizedReRoPE(LlamaAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device

        assert isinstance(past_key_value, VectorQuantizedReRoPECache), "VQReRoPECache is required"
        assert (
            past_key_value.get_seq_length(self.layer_idx) == 0 or q_len == 1
        ), "Only support prefilling or generation"  # TODO
        assert not output_attentions, "Not implemented"
        assert not self.training, "Not implemented"
        assert not past_key_value.sphere, "Not implemented"

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        attn_output = self._dispatch_attn_impl(
            query_states, key_states, value_states, position_ids, kv_seq_len, past_key_value
        )

        # attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _dispatch_attn_impl(
        self,
        query_states,
        key_states,
        value_states,
        position_ids,
        kv_seq_len,
        past_key_value: VectorQuantizedReRoPECache,
    ):

        is_prefilling = past_key_value.get_seq_length(self.layer_idx) == 0
        is_prefilling_with_memory = is_prefilling and query_states.shape[2] > past_key_value.window_length

        inputs = (query_states, key_states, value_states, position_ids, kv_seq_len, past_key_value)
        if is_prefilling_with_memory:
            if past_key_value.attention_mode == AttentionMode.SORT_BASED_LINEAR:
                return self.long_prefilling_sort_based_linear(*inputs)
            elif past_key_value.attention_mode == AttentionMode.SLOW_REFERNECE:
                return self.long_prefilling_slow_reference_impl(*inputs)
            raise KeyError(f"Bad attention mode: {past_key_value.attention_mode}")
        else:
            return self.naive_attn(query_states, key_states, value_states, position_ids, kv_seq_len, past_key_value)

    def _type_guard(self, vectors: list[torch.Tensor] | torch.Tensor):

        if isinstance(vectors, torch.Tensor):
            vectors = [vectors]
            return_single = True
        else:
            return_single = False

        input_dtype = vectors[0].dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            vectors = [item.to(target_dtype) for item in vectors]

        return vectors[0] if return_single else vectors

    def long_prefilling_slow_reference_impl(
        self,
        query_states,
        key_states,
        value_states,
        position_ids,
        kv_seq_len,
        past_key_value: VectorQuantizedReRoPECache,
    ):

        logger.warning_once("This is only for checking the correctness of linear-time prefilling.")

        bsz, hd, q_len, hidden_size = query_states.shape

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states1, key_states1 = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        w_position = torch.full_like(position_ids, past_key_value.window_length - 1)
        cos_at_w, sin_at_w = cos[w_position].unsqueeze(1), sin[w_position].unsqueeze(1)
        query_states2 = (query_states * cos_at_w) + (rotate_half(query_states) * sin_at_w)
        key_states2, vq_key_ids, vq_key_onehot = multi_head_single_quantize(key_states, past_key_value.codebook)

        output = torch.zeros_like(query_states, memory_format=torch.contiguous_format)
        BLOCK_SIZE = 128
        WIN = past_key_value.window_length
        SOFTMAX_SCALE = 1.0 / math.sqrt(hidden_size)
        for _b in range(bsz):
            for _h in range(hd):
                for qi in range(q_len):
                    boundary = max(0, (qi - WIN) // BLOCK_SIZE * BLOCK_SIZE)
                    q1 = query_states1[_b, _h, qi]
                    k1 = key_states1[_b, _h, boundary : qi + 1]
                    v = value_states[_b, _h, max(0, (qi - WIN) // BLOCK_SIZE * BLOCK_SIZE) : qi + 1]
                    attn = q1 @ k1.T
                    if boundary > 0:
                        q2 = query_states2[_b, _h, qi]
                        k2 = key_states2[_b, _h, :boundary]
                        v2 = value_states[_b, _h, :boundary]
                        attn2 = q2 @ k2.T
                        attn = torch.cat([attn, attn2], dim=0)
                        v = torch.cat([v, v2], dim=0)
                    attn = torch.softmax(attn.float() * SOFTMAX_SCALE, dim=0).to(v.dtype)
                    o = attn @ v
                    output[_b, _h, qi] = o
        return output.transpose(1, 2).contiguous().flatten(2)

    def long_prefilling_sort_based_linear(
        self,
        query_states,
        key_states,
        value_states,
        position_ids,
        kv_seq_len,
        past_key_value: VectorQuantizedReRoPECache,
    ):
        bsz, hd, q_len, hidden_size = query_states.shape
        cdsz = past_key_value.codebook.shape[1]  # codebook size
        device = query_states.device
        dtype = query_states.dtype

        # vq_key_ids: batch x n_head x seqlen
        vq_keys, vq_key_ids, vq_key_onehot = multi_head_single_quantize(key_states, past_key_value.codebook)
        vq_keys_cumsum = torch.cumsum(vq_key_onehot, dim=2)

        # !assume max(vq_key_ids) < 2^13,
        # !assume q_len < 2^18
        vq_key_ids_int32 = vq_key_ids.to(torch.int32)
        pos_arange = torch.arange(q_len, dtype=torch.int32, device=device)
        vq_key_id_and_pos = (vq_key_ids_int32 << 18) + pos_arange[None, None, :]

        sorted_ids = vq_key_id_and_pos.argsort()
        _ids = sorted_ids[..., None].expand(-1, -1, -1, self.head_dim)
        # 0, [C1], [C2] ... [Cn]
        vq_value_states = torch.zeros(
            (bsz, self.num_key_value_heads, q_len + 1, self.head_dim), device=device, dtype=dtype
        )

        # value_states = vq_key_ids_int32.unsqueeze(-1).expand(-1, -1, -1, 64)
        torch.gather(value_states, 2, _ids, out=vq_value_states[:, :, 1:])  # batch x n_head x seqlen x head_dim

        num_in_unrolled = vq_keys_cumsum[:, :, -1]
        boundary_in_unrolled = torch.cumsum(num_in_unrolled, dim=2)
        boundary_in_unrolled = torch.roll(boundary_in_unrolled, 1)
        boundary_in_unrolled[..., 0] = 0
        vq_value_states_orig = vq_value_states
        vq_value_states = torch.cumsum(vq_value_states.float(), 2)
        boundary_vec = vq_value_states.gather(2, boundary_in_unrolled.unsqueeze(-1).expand(-1, -1, -1, hidden_size))
        # NOTE torch.repeat_interleave cannot be batched by torch.vmap
        boundary_vec = torch.vmap(torch.vmap(partial(torch.repeat_interleave, dim=0)))(boundary_vec, num_in_unrolled)

        vq_value_states[:, :, 1:] -= boundary_vec
        vq_value_states = vq_value_states.to(dtype)
        vq_value_indices = vq_keys_cumsum + boundary_in_unrolled[:, :, None]

        # for _b in range(bsz):
        #     for _h in range(hd):
        #         for qi in range(32, q_len):
        #             debug_v = torch.zeros(64, device='cuda')
        #             for ci in range(cdsz):
        #                 if vq_keys_cumsum[_b, _h, qi, ci] == 0:
        #                     continue
        #                 i = vq_value_indices[_b, _h, qi, ci]
        #                 v = vq_value_states[_b, _h, i]
        #                 mask = vq_key_ids[_b, _h, : qi + 1] == ci
        #                 av = value_states[_b, _h, : qi + 1][mask]
        #                 av = av.sum(0)
        #                 debug_v += av
        #             if qi == 127:
        #                 print(debug_v)
        #                 breakpoint()
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states1, key_states1 = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        w_position = torch.full_like(position_ids, past_key_value.window_length - 1)
        cos_at_w, sin_at_w = cos[w_position].unsqueeze(1), sin[w_position].unsqueeze(1)
        query_states2 = (query_states * cos_at_w) + (rotate_half(query_states) * sin_at_w)

        # grouping
        BLOCK_SIZE = 128  # MUST MATCH KERNEL
        vq_keys_cumsum_all = vq_keys_cumsum
        vq_keys_cumsum = vq_keys_cumsum.view(bsz, hd, -1, BLOCK_SIZE, cdsz)[:, :, :, -1].contiguous()
        vq_value_indices_all = vq_value_indices
        vq_value_indices = vq_value_indices.view(bsz, hd, -1, BLOCK_SIZE, cdsz)[:, :, :, -1].contiguous()
        # TODO value can be grouped as well. optimize later

        # Z, H, N_CTX, D_HEAD
        query_states = self._type_guard((query_states1, query_states2))
        key_states = self._type_guard((key_states1, past_key_value.codebook, vq_keys_cumsum.to(dtype)))
        value_states = self._type_guard((value_states, vq_value_states, vq_value_indices))
        out = vq_prefilling_kernel(
            *query_states,
            *key_states,
            *value_states,
            past_key_value.window_length,
            softmax_scale=1 / math.sqrt(self.head_dim),
        )
        out = out.transpose(1, 2).contiguous().flatten(2)
        return out

    def naive_attn(
        self,
        query_states,
        key_states,
        value_states,
        position_ids,
        kv_seq_len,
        past_key_value: VectorQuantizedReRoPECache,
    ):
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Z, N_CTX, H, D_HEAD
        query_states: torch.Tensor
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        query_states, key_states, value_states = self._type_guard([query_states, key_states, value_states])

        # (batch_size, seqlen, (nheads, headdim))
        return (
            naive_prefilling_kernel(
                query_states,
                key_states,
                value_states,
                softmax_scale=1 / math.sqrt(self.head_dim),
                causal=True,
            )
            .contiguous()
            .flatten(2)
        )


@contextmanager
def use_rerope():
    LLAMA_ATTENTION_CLASSES["flash_attention_2"] = LlamaFlashAttention2VectorQuantizedReRoPE
    yield
    LLAMA_ATTENTION_CLASSES["flash_attention_2"] = LlamaFlashAttention2


# if __name__ == "__main__":
#     B, H, L, D = 1, 2, 10, 3
#     C = 7
#     q_len = L
#     device = "cpu"

#     codebook = torch.randn(H, C, D)
#     key_states = torch.randn(B, H, L, D)
#     value_states = torch.randn(B, H, L, D)

#     # vq_key_ids: batch x n_head x seqlen
#     vq_keys, vq_key_ids, vq_key_onehot = multi_head_single_quantize(key_states, codebook)
#     vq_keys_cumsum = torch.cumsum(vq_key_onehot, dim=2)

#     # EXPERIMENTAL IMPL 1 - SORTING BASED

#     # !assume max(vq_key_ids) < 2^13,
#     # !assume q_len < 2^18

#     vq_key_ids_int32 = vq_key_ids.to(torch.int32)
#     pos_arange = torch.arange(q_len, dtype=torch.int32, device=device)
#     vq_key_id_and_pos = (vq_key_ids_int32 << 18) + pos_arange[None, None, :]

#     sorted_ids = vq_key_id_and_pos.argsort()
#     _ids = sorted_ids[..., None].expand(-1, -1, -1, D)
#     # 0, [C1], [C2] ... [Cn]
#     vq_value_states = torch.zeros((B, H, q_len + 1, D), device=device)
#     torch.gather(value_states, 2, _ids, out=vq_value_states[:, :, 1:])  # batch x n_head x seqlen x head_dim
#     # _debug_ids = vq_key_ids.gather(2, sorted_ids)

#     # cumsum in each code
#     vq_value_states.cumsum_(2)
#     # following codes can be implemented in Triton in a much clear and faster way.
#     nums1 = torch.sum(vq_key_onehot, dim=2)
#     nums = torch.cumsum(nums1, dim=2)
#     boundaries = torch.roll(nums, 1)
#     boundaries[..., 0] = 0
#     boundary_vec = vq_value_states.gather(2, boundaries.unsqueeze(-1).expand(-1, -1, -1, D))
#     # NOTE torch.repeat_interleave cannot be batched by torch.vmap
#     boundary_vec = torch.vmap(torch.vmap(partial(torch.repeat_interleave, dim=0)))(boundary_vec, nums1)
#     vq_value_states[:, :, 1:] -= boundary_vec

#     vq_value_indices = vq_keys_cumsum + boundaries[:, :, None]


if __name__ == "__main__":
    from transformers.models.llama import LlamaConfig

    config = LlamaConfig(hidden_size=128, num_attention_heads=2, num_key_value_heads=2)
    module = LlamaFlashAttention2VectorQuantizedReRoPE(config, layer_idx=0).half().cuda()
    module.eval()

    with torch.no_grad():
        codebook = torch.randn(2, 128, 64, dtype=torch.float16, device="cuda")
        x = torch.randn(1, 512, 128, dtype=torch.float16, device="cuda")  # bsz, qlen, hidden
        position_ids = torch.arange(512, device="cuda").unsqueeze(0)  # bsz, length

        cache = VectorQuantizedReRoPECache(256, codebook, False, attention_mode=AttentionMode.SORT_BASED_LINEAR)
        o_fast = module(hidden_states=x, position_ids=position_ids, past_key_value=cache)[0]

        cache = VectorQuantizedReRoPECache(256, codebook, False, attention_mode=AttentionMode.SLOW_REFERNECE)
        o_ref = module(hidden_states=x, position_ids=position_ids, past_key_value=cache)[0]

        print((o_ref[0, 0] - o_fast[0, 0]).abs().max())
        print(o_ref[0, 0])
        print(o_fast[0, 0])
        breakpoint()
