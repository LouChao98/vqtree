# A minimal implementation of VQ in Transformer for inference only


import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as distributed
from torch.cuda.amp import autocast

from einops import rearrange, repeat, reduce, pack, unpack


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def identity(t):
    return t


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def cdist(x, y):
    x2 = reduce(x**2, "b n d -> b n", "sum")
    y2 = reduce(y**2, "b n d -> b n", "sum")
    xy = einsum("b i d, b j d -> b i j", x, y) * -2
    return (rearrange(x2, "b i -> b i 1") + rearrange(y2, "b j -> b 1 j") + xy).sqrt()


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def laplace_smoothing(x, n_categories, eps=1e-5, dim=-1):
    denom = x.sum(dim=dim, keepdim=True)
    return (x + eps) / (denom + n_categories * eps)


def pad_shape(shape, size, dim=0):
    return [size if i == dim else s for i, s in enumerate(shape)]


def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype=dtype, device=device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target


def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, "h b n -> h b n d", d=dim)
    embeds = repeat(embeds, "h c d -> h b c d", b=batch)
    return embeds.gather(2, indices)


# distance types


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks=1,
        eps=1e-5,
    ):
        super().__init__()
        self.transform_input = identity

        embed = torch.empty(num_codebooks, codebook_size, dim)
        self.register_buffer("embed", embed)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.eps = eps

    @autocast(enabled=False)
    def forward(self, x, mask=None):
        x = x.float()
        flatten, ps = pack_one(x, "h * d")

        if exists(mask):
            mask = repeat(
                mask,
                "b n -> c (b h n)",
                c=flatten.shape[0],
                h=flatten.shape[-2] // (mask.shape[0] * mask.shape[1]),
            )

        dist = -cdist(flatten, self.embed)

        embed_ind = dist.argmax(dim=-1)
        embed_ind = unpack_one(embed_ind, ps, "h *")

        quantize = batched_embedding(embed_ind, self.embed)

        dist = unpack_one(dist, ps, "h * d")
        return quantize, embed_ind, dist


class CosineSimCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks=1,
        eps=1e-5,
        learnable_codebook=False,
    ):
        super().__init__()
        self.transform_input = l2norm

        embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        self.register_buffer("embed", embed)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.eps = eps

    @autocast(enabled=False)
    def forward(self, x, mask=None):
        needs_codebook_dim = x.ndim < 4

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, "... -> 1 ...")

        dtype = x.dtype

        flatten, ps = pack_one(x, "h * d")

        if exists(mask):
            mask = repeat(
                mask,
                "b n -> c (b h n)",
                c=flatten.shape[0],
                h=flatten.shape[-2] // (mask.shape[0] * mask.shape[1]),
            )

        embed = self.embed if self.learnable_codebook else self.embed.detach()

        dist = einsum("h n d, h c d -> h n c", flatten, embed)

        embed_ind = dist.argmax(dim=-1)
        embed_ind = unpack_one(embed_ind, ps, "h *")

        quantize = batched_embedding(embed_ind, embed)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, "1 ... -> ..."), (quantize, embed_ind))

        dist = unpack_one(dist, ps, "h * d")
        return quantize, embed_ind, dist


# main class


class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        heads=1,
        separate_codebook_per_head=False,
        eps=1e-5,
        use_cosine_sim=False,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        self.eps = eps

        codebook_class = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook

        codebook_kwargs = dict(
            dim=dim,
            num_codebooks=heads if separate_codebook_per_head else 1,
            codebook_size=codebook_size,
            eps=eps,
        )

        self._codebook = codebook_class(**codebook_kwargs)

        self.codebook_size = codebook_size

    @property
    def codebook(self):
        codebook = self._codebook.embed

        if self.separate_codebook_per_head:
            return codebook

        return rearrange(codebook, "1 ... -> ...")

    @codebook.setter
    def codebook(self, codes):
        if not self.separate_codebook_per_head:
            codes = rearrange(codes, "... -> 1 ...")

        self._codebook.embed.copy_(codes)

    def get_codes_from_indices(self, indices):
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            codes = codebook[indices]
            return rearrange(codes, "... h d -> ... (h d)")

        indices, ps = pack_one(indices, "b * h")
        indices = rearrange(indices, "b n h -> b h n")

        indices = repeat(indices, "b h n -> b h n d", d=codebook.shape[-1])
        codebook = repeat(codebook, "h n d -> b h n d", b=indices.shape[0])

        codes = codebook.gather(2, indices)
        codes = rearrange(codes, "b h n d -> b n (h d)")
        codes = unpack_one(codes, ps, "b * d")
        return codes

    def get_output_from_indices(self, indices):
        return self.get_codes_from_indices(indices)

    def forward(self, x, mask=None):
        orig_input = x

        only_one = x.ndim == 2

        if only_one:
            assert not exists(mask)
            x = rearrange(x, "b d -> b 1 d")

        heads, is_multiheaded = self.heads, self.heads > 1

        # handle multi-headed separate codebooks

        if is_multiheaded:
            ein_rhs_eq = "h b n d" if self.separate_codebook_per_head else "1 (b h) n d"
            x = rearrange(x, f"b n (h d) -> {ein_rhs_eq}", h=heads)

        # l2norm for cosine sim, otherwise identity

        x = self._codebook.transform_input(x)

        # quantize

        quantize, embed_ind, distances = self._codebook(x, mask=mask)

        # transform embedding indices

        if is_multiheaded:
            if self.separate_codebook_per_head:
                embed_ind = rearrange(embed_ind, "h b n -> b n h", h=heads)
            else:
                embed_ind = rearrange(embed_ind, "1 (b h) n -> b n h", h=heads)

        if only_one:
            embed_ind = rearrange(embed_ind, "b 1 ... -> b ...")

        # handle multi-headed quantized embeddings

        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, "h b n d -> b n (h d)", h=heads)
            else:
                quantize = rearrange(quantize, "1 (b h) n d -> b n (h d)", h=heads)

        # rearrange quantized embeddings

        if only_one:
            quantize = rearrange(quantize, "b 1 d -> b d")

        # if masking, only return quantized for where mask has True

        if exists(mask):
            quantize = torch.where(rearrange(mask, "... -> ... 1"), quantize, orig_input)

        return quantize, embed_ind
