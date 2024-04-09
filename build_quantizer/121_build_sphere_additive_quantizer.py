import os

os.environ["OMP_NUM_THREADS"] = "12"

import faiss
import numpy as np
from pathlib import Path
from faiss.contrib.inspect_tools import get_additive_quantizer_codebooks
import pickle
from concurrent.futures import ThreadPoolExecutor

# 512 * 4
params = {"num_codebooks": 4, "codebook_bits": 9, "max_beam_size": 6, "use_beam_LUT": 1}
source = Path("/home/louchao/vqtree/data/extracted/LWM_SP6B-len4096-num100-seed42.npy")


def eval_codec(q, xb):
    codes = q.compute_codes(xb)
    decoded = q.decode(codes)
    return ((xb - decoded) ** 2).sum()


def train_quantizer(data):
    rq0 = faiss.ResidualQuantizer(data.shape[-1], params["num_codebooks"], params["codebook_bits"])
    rq0.train_type = faiss.ResidualQuantizer.Train_progressive_dim
    rq0.max_beam_size = params["max_beam_size"]
    rq0.use_beam_LUT = params["use_beam_LUT"]

    rq0.train(data)

    cb = get_additive_quantizer_codebooks(rq0)
    return cb


origin_data = np.load(source, mmap_mode="r")
output = {}


with ThreadPoolExecutor(max_workers=5) as pool:
    for head_idx in range(origin_data.shape[1]):

        futures = []
        for layer_idx in range(origin_data.shape[0]):
            data = origin_data[layer_idx, head_idx].copy()
            data = data.reshape(-1, data.shape[-1])
            data /= np.linalg.norm(data, axis=1, keepdims=True)
            future = pool.submit(train_quantizer, data)
            futures.append((layer_idx, head_idx, future))

        for layer_idx, head_idx, future in futures:
            print(layer_idx, head_idx)
            output[(layer_idx, head_idx)] = future.result()

        # np.random.seed(42)
        # sampled_data = np.random.choice(data.shape[0], size=50, replace=False)
        # test_data = data[sampled_data]

        # err = eval_codec(rq0, test_data)
        # print(err)

output_path = source.parent / source.stem
output_path.mkdir(exist_ok=True)
with open(
    output_path / "sphere_ncb{num_codebooks}-bits{codebook_bits}-mbsz{max_beam_size}-lut{use_beam_LUT}.pkl".format(**params),
    "wb",
) as f:
    pickle.dump(output, f)
