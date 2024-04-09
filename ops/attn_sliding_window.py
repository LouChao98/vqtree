import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch
from flash_attn import flash_attn_func
from transformers import SinkCache
from transformers.generation.utils import Cache
from transformers.models.llama.modeling_llama import (LLAMA_ATTENTION_CLASSES,
                                                      LlamaAttention,
                                                      LlamaFlashAttention2,
                                                      apply_rotary_pos_emb,
                                                      logger, rotate_half)


class LlamaFlashAttention2SlidingWindow(LlamaAttention):

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

        assert isinstance(past_key_value, SinkCache), "SinkCache is required"
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

        if past_key_value.get_seq_length() == 0:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=1)
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

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
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
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            softmax_scale=softmax_scale,
            window_size=(window, -1),
            causal=True,
        )
        return attn_output


@contextmanager
def use_sliding_window():
    LLAMA_ATTENTION_CLASSES["flash_attention_2"] = LlamaFlashAttention2SlidingWindow
    yield
    LLAMA_ATTENTION_CLASSES["flash_attention_2"] = LlamaFlashAttention2
