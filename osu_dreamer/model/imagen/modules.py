import math
from functools import partial

import torch
from torch import nn

from xformers.ops import memory_efficient_attention

from einops import rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class LayerNorm(nn.Module):
    def __init__(self, feats, stable=False, dim=-1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim

        if self.stable:
            x = x / x.amax(dim=dim, keepdim=True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=dim, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=dim, keepdim=True)

        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


def Downsample(dim, dim_out=None):
    dim_out = dim if dim_out is None else dim_out
    return nn.Conv1d(dim, dim_out, 4, 2, 1, padding_mode="reflect")


def Upsample(dim, dim_out=None):
    dim_out = dim if dim_out is None else dim_out
    return nn.ConvTranspose1d(dim, dim_out, 4, 2, 1)


class LearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with learned sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# only implement flash attention for now
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads=8,
        dim_head=64,
    ):
        super().__init__()
        self.scale = dim_head**-0.5

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim),
        )

    def forward(self, x):
        x = rearrange(x, "b c n -> b n c")
        b = x.shape[0]

        x = self.norm(x)

        q = rearrange(self.to_q(x), "b n (h d) -> b n h d", h=self.heads)
        kv = rearrange(
            self.to_kv(x), "b n (two h d) -> b n two h d", two=2, h=self.heads
        )

        k, v = kv.unbind(dim=-3)

        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
        out = memory_efficient_attention(q, k, v, scale=self.scale)

        out = rearrange(out, "b n h d -> b n (h d)", b=b, h=self.heads)
        out = self.to_out(out.to(x.dtype))
        out = rearrange(out, "b n c -> b c n")
        return out


class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups=8,
        norm=True,
    ):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, dim) if norm else Identity()
        self.activation = nn.SiLU()
        self.project = nn.Conv1d(dim, dim_out, 7, padding=3)

    def forward(self, x, scale_shift=None):
        x = self.group_norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        x = self.project(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        time_cond_dim=None,
        groups=8,
    ):
        super().__init__()

        self.time_mlp = None
        if time_cond_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2),
            )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        self.res_conv = (
            nn.Conv1d(dim, dim_out, 7, padding=3, padding_mode="reflect")
            if dim != dim_out
            else Identity()
        )

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x)
        h = self.block2(h, scale_shift=scale_shift)

        return h + self.res_conv(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class UNet(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        h_dims,
        h_dim_groups,
        blocks_per_depth,
        attn_heads,
        attn_dim,
        learned_sinu_pos_emb_dim,
    ):
        super().__init__()

        block = partial(ResnetBlock, groups=h_dim_groups)

        in_out = list(zip(h_dims[:-1], h_dims[1:]))
        num_layers = len(in_out)

        self.init_conv = nn.Sequential(
            nn.Conv1d(in_dim, h_dims[0], 7, padding=3),
        )

        # time embeddings
        emb_dim = h_dims[0] * 4
        self.time_mlp = nn.Sequential(
            LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim),
            nn.Linear(learned_sinu_pos_emb_dim + 1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # layers

        self.downs = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                block(
                                    dim_in if i == 0 else dim_out,
                                    dim_out,
                                    time_cond_dim=emb_dim,
                                )
                                for i in range(blocks_per_depth)
                            ]
                        ),
                        nn.ModuleList(
                            [
                                Residual(
                                    PreNorm(
                                        dim_out,
                                        Attention(
                                            dim_out, heads=attn_heads, dim_head=attn_dim
                                        ),
                                    )
                                )
                                for _ in range(blocks_per_depth)
                            ]
                        ),
                        Downsample(dim_out)
                        if ind < (num_layers - 1)
                        else nn.Identity(),
                    ]
                )
                for ind, (dim_in, dim_out) in enumerate(in_out)
            ]
        )

        mid_dim = h_dims[-1]
        self.mid_block1 = block(mid_dim, mid_dim, time_cond_dim=emb_dim)
        self.mid_attn = Residual(
            PreNorm(mid_dim, Attention(mid_dim, heads=attn_heads, dim_head=attn_dim))
        )
        self.mid_block2 = block(mid_dim, mid_dim, time_cond_dim=emb_dim)

        self.ups = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                block(
                                    dim_out * 2 if i == 0 else dim_in,
                                    dim_in,
                                    time_cond_dim=emb_dim,
                                )
                                for i in range(blocks_per_depth)
                            ]
                        ),
                        nn.ModuleList(
                            [
                                Residual(
                                    PreNorm(
                                        dim_in,
                                        Attention(
                                            dim_in, heads=attn_heads, dim_head=attn_dim
                                        ),
                                    )
                                )
                                for _ in range(blocks_per_depth)
                            ]
                        ),
                        Upsample(dim_in) if ind < (num_layers - 1) else nn.Identity(),
                    ]
                )
                for ind, (dim_in, dim_out) in enumerate(in_out[::-1])
            ]
        )

        self.final_conv = nn.Sequential(
            *(block(h_dims[0], h_dims[0]) for _ in range(blocks_per_depth)),
            zero_module(nn.Conv1d(h_dims[0], out_dim, 1)),
        )

    def forward(self, x, a, ts):

        x = torch.cat([x, a], dim=1)

        x = self.init_conv(x)

        h = []
        emb = self.time_mlp(ts)

        # downsample
        for blocks, attns, downsample in self.downs:
            for block, attn in zip(blocks, attns):
                x = attn(block(x, emb))
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        # upsample
        for blocks, attns, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            for block, attn in zip(blocks, attns):
                x = attn(block(x, emb))
            x = upsample(x)

        return self.final_conv(x)
