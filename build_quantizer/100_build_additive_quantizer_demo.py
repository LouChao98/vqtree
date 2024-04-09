import faiss
import numpy as np
import nni

from faiss.contrib.inspect_tools import get_additive_quantizer_codebooks

rq0 = faiss.ResidualQuantizer(128, 3, 9)
rq0.train_type = faiss.ResidualQuantizer.Train_progressive_dim
rq0.max_beam_size = 8
rq0.use_beam_LUT = 1

xt = np.random.randn(10000, 128)
rq0.train(xt)
cb0 = get_additive_quantizer_codebooks(rq0)

