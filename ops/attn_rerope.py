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
from flash_attn import flash_attn_func
from ops.triton_fused_attn_rerope import attention
from contextlib import contextmanager


class ReRoPECache(Cache):
    # A cache that as described in https://kexue.fm/archives/9708
    # Based on SinkCache

    def __init__(
        self,
        window_length: int,
    ) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.window_length = window_length
        self.cos_1 = {}
        self.sin_1 = {}
        self.seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

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


class LlamaFlashAttention2ReRoPE(LlamaAttention):

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

        assert isinstance(past_key_value, ReRoPECache), "ReRoPECache is required"
        assert (
            past_key_value.get_seq_length(self.layer_idx) == 0 or q_len == 1
        ), "Only support prefilling or generation"
        assert not output_attentions, "Not implemented"
        assert not self.training, "Not implemented"
        assert bsz == 1, "Not implemented"

        is_prefilling = past_key_value.get_seq_length(self.layer_idx) == 0

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

        if is_prefilling and q_len > past_key_value.window_length:

            # prefilling with extended ctx

            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

            query_states1, key_states1 = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            # w_position = torch.full_like(position_ids, past_key_value.window_length)
            # cos_at_w, sin_at_w = cos[w_position].unsqueeze(1), sin[w_position].unsqueeze(1)
            # query_states2 = (query_states * cos_at_w) + (rotate_half(query_states) * sin_at_w)
            key_states2 = key_states

            query_states2, _ = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, torch.tensor([past_key_value.window_length], device=device)
            )
            _, key_states2 = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, torch.tensor([0], device=device)
            )
            _, key_states3 = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                torch.tensor([q_len - past_key_value.window_length], device=device),
            )

            # m1 = (query_states1[0, 0, :, 0][:, None] - key_states1[0, 0, :, 0][None, :]).tril()
            # # breakpoint()
            # m2 = (query_states2[0, 0, :, 0][:, None] - key_states2[0, 0, :, 0][None, :]).tril()
            # m = torch.where(m1<past_key_value.window_length, m1, m2)
            # breakpoint()

            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            merged_key_states = torch.cat(
                [
                    key_states3[:, :, : -past_key_value.window_length],
                    key_states1[:, :, -past_key_value.window_length :],
                ],
                dim=2,
            )
            past_key_value.update(merged_key_states, value_states, self.layer_idx, cache_kwargs)

            # Z, H, N_CTX, D_HEAD
            query_states = (query_states1, query_states2)
            key_states = (key_states1, key_states2)
        else:

            # generation or prefilling with short ctx
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # Z, N_CTX, H, D_HEAD
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = value_states.dtype
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

            if isinstance(query_states, tuple):
                query_states = (query_states[0].to(target_dtype), query_states[1].to(target_dtype))
                key_states = (key_states[0].to(target_dtype), key_states[1].to(target_dtype))
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            window=past_key_value.window_length,
            softmax_scale=1 / math.sqrt(self.head_dim),
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        window,
        softmax_scale,
    ):
        if isinstance(query_states, tuple):
            # Z, H, N_CTX, D_HEAD
            attn_output = attention(
                query_states[0],
                query_states[1],
                key_states[0],
                key_states[1],
                value_states,
                True,
                softmax_scale,
                window,
            )
            attn_output = attn_output.transpose(1, 2)
        else:
            # (batch_size, seqlen, nheads, headdim)
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                softmax_scale=softmax_scale,
                causal=True,
            )
        return attn_output


@contextmanager
def use_rerope():
    LLAMA_ATTENTION_CLASSES["flash_attention_2"] = LlamaFlashAttention2ReRoPE
    yield
    LLAMA_ATTENTION_CLASSES["flash_attention_2"] = LlamaFlashAttention2
