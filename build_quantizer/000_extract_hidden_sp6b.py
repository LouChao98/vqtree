import argparse
import random
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_dataset import LMDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    help="Path to pretrained model",
    default="/home/louchao/vqtree/data/LWM-Text-Chat-1M",
)
parser.add_argument(
    "--dataset",
    type=str,
    help="Path to tokenized dataset",
    default="/home/louchao/rdt/training/data/slimpajama-6b/cache/tokenizer_name-sheared-llama-1.3b-val_ratio-0.0005-val_split_seed-2357-add_eos-True-detokenize-False/train.npy",
)
parser.add_argument("--seqlen", type=int, help="Length for each input", default=4096)
parser.add_argument("--numseq", type=int, help="Number of sentences", default=100)
parser.add_argument("--batch_size", type=int, help="Batch size for extraction", default=2)
parser.add_argument("--out_name", type=str, help="output file name", default="extracted")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
)
model.eval()

dataset = LMDataset(np.load(args.dataset, mmap_mode="r"), args.seqlen)
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1, shuffle=True)

store = np.empty(
    (
        model.config.num_hidden_layers,
        model.config.num_key_value_heads,
        args.numseq,
        args.seqlen,
        model.config.hidden_size // model.config.num_attention_heads,
    ),
    dtype=np.float16,
)
cursor = 0

with torch.inference_mode():
    for x, y in tqdm(dataloader, total=(args.numseq + args.batch_size - 1) // args.batch_size):
        if cursor >= args.numseq:
            break

        output = model(x, use_cache=True, return_dict=True)["past_key_values"]

        for li, (keys, _) in enumerate(output):

            batch_size = keys.shape[0]
            keys = keys.permute(1, 0, 2, 3).cpu().numpy()
            store[li, :, cursor : cursor + batch_size] = keys[:, : min(batch_size, args.numseq - cursor)]

        cursor += batch_size

np.save(f"../data/extracted/{args.out_name}-len{args.seqlen}-num{args.numseq}-seed{args.seed}.npy", store)
