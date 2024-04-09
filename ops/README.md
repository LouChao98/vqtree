
### Standard attention

* `triton_fused_attn_orig.py`: the fwd impl in Tri Dao's flash-attn repo.
* `triton_fused_attn.py`: minor modification to run on the latest Triton
* `triton_fused_attn2.py`: use new APIs in the latest Triton

Use `triton_fused_attn2.py`.

### ReRoPE

* `triton_fused_attn_rerope_orig.py`: the fwd impl in https://gist.github.com/chu-tianxiang/4307937fd94b49c75b61a6967716bae9. minor modification.
* `triton_fused_attn_rerope.py`: more modification on boundary check and dtype.

Use `triton_fused_attn_rerope.py`.

### Local

* `triton_fused_local_attn.py` sliding window attn
* `triton_fused_local_attn2.py` sliding window attn with loose mask. (block-wise)

### ReRoPE + VQ

* `triton_fused_attn_vq.py`