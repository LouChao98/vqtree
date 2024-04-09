from transformers.generation.utils import Cache
import transformers
import transformers.models.llama.modeling_llama


class FlattenVQCache(Cache):
    
    ...
    
    
def replace_llama_attn():
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask
    