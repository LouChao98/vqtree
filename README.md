conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.2/

HF_DATASETS_OFFLINE=1 HF_DATASETS_OFFLINE=1

### Note

* Most kernels implement the flash-attention v2.0's behavior on the masking of seqlen_q != seqlen_k. But we only consider prefilling (seqlen_q = seqlen_k) and generation (seqlen_q = 1), and the latter is handled by a seperate kernel.

* rerope and standard attn seems not to use the flash-attn2's non-matmul optimization. TODO implement it
* rerope and standard attn seems not to use the log2 optimization. TODO implement it