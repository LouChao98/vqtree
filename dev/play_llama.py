from transformers.models.llama import LlamaForCausalLM
from transformers.generation.utils import GenerationMixin


GenerationMixin.greedy_search


# mini preparing in greedy search:

LlamaForCausalLM._prepare_model_inputs
LlamaForCausalLM._prepare_attention_mask_for_generation

# mini loop in greedy search:

LlamaForCausalLM.prepare_inputs_for_generation(...)

LlamaForCausalLM.forward(...)

LlamaForCausalLM._update_model_kwargs_for_generation(...)
..., LlamaForCausalLM._extract_past_from_model_output

LlamaForCausalLM.generate