from transformers import AutoTokenizer
from transformers import SinkCache
import transformers
import torch
from ops.attn_rerope import ReRoPECache, use_rerope
from ops.attn_sliding_window import use_sliding_window

model = "/home/louchao/vqtree/data/sheared-llama-1.3b"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline_kwargs = {
    "task": "text-generation",
    "model": model,
    "torch_dtype": torch.float16,
    "device_map": "auto",
    "model_kwargs": {"attn_implementation": "flash_attention_2"},
}


def run_gen(pipeline, **kwargs):
    sequences = pipeline(
        'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=32,
        truncation=True,
        **kwargs,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")


# print("=============== naive ===============")
# pipeline = transformers.pipeline(**pipeline_kwargs)
# run_gen(pipeline)

print("=============== sliding window ===============")
with use_sliding_window():
    pipeline = transformers.pipeline(**pipeline_kwargs)
run_gen(pipeline, past_key_values=SinkCache(window_length=16, num_sink_tokens=0))

# print("=============== attention sink===============")
# pipeline = transformers.pipeline(**pipeline_kwargs)
# run_gen(pipeline)

print("=============== rerope ===============")
with use_rerope():
    pipeline = transformers.pipeline(**pipeline_kwargs)
run_gen(pipeline, past_key_values=ReRoPECache(16))
