import os

os.environ["OMP_NUM_THREADS"] = "8"

import faiss
import numpy as np
import nni


params = nni.get_next_parameter()
# params = {"num_codebooks": 3, "codebook_bits": 8, "max_beam_size": 5, "use_beam_LUT": 1}


def eval_codec(q, xb):
    codes = q.compute_codes(xb)
    decoded = q.decode(codes)
    return ((xb - decoded) ** 2).sum()


data = np.load("/home/louchao/vqtree/data/extracted/LWM_SP6B-len4096-num100-seed42.npy", mmap_mode="r")
data = data[0, 0].copy()
data = data.reshape(-1, data.shape[-1])
print(data.shape)

rq0 = faiss.ResidualQuantizer(data.shape[-1], params["num_codebooks"], params["codebook_bits"])
rq0.train_type = faiss.ResidualQuantizer.Train_progressive_dim
rq0.max_beam_size = params["max_beam_size"]
rq0.use_beam_LUT = params["use_beam_LUT"]

rq0.train(data)

np.random.seed(42)
sampled_data = np.random.choice(data.shape[0], size=50, replace=False)
test_data = data[sampled_data]

err = eval_codec(rq0, test_data)
print(err)
nni.report_final_result(err)