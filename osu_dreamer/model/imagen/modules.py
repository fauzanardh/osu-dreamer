import math
from functools import partial

import torch
import torch.nn as nn
from xformers.ops import memory_efficient_attention

from einops import rearrange


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    return nn.ConvTranspose1d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv1d(dim, dim, 4, 2, 1, padding_mode="reflect")


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


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


class WaveBlock(nn.Module):
    """context is acquired from num_stacks*2**stack_depth neighborhood"""

    def __init__(self, dim, stack_depth, num_stacks, mult=1, h_dim_groups=1, up=False):
        super().__init__()

        self.in_net = nn.Conv1d(dim, dim * mult, 1)
        self.out_net = nn.Conv1d(dim * mult, dim, 1)

        self.nets = nn.ModuleList(
            [
                nn.Sequential(
                    (nn.ConvTranspose1d if up else nn.Conv1d)(
                        in_channels=dim * mult,
                        out_channels=2 * dim * mult,
                        kernel_size=2,
                        padding=2**i,
                        dilation=2 ** (i + 1),
                        groups=h_dim_groups,
                        **({} if up else dict(padding_mode="replicate")),
                    ),
                    nn.GLU(dim=1),
                )
                for _ in range(num_stacks)
                for i in range(stack_depth)
            ]
        )

    def forward(self, x):
        x = self.in_net(x)
        h = x
        for net in self.nets:
            h = net(h)
            x = x + h
        return self.out_net(x)


class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, emb_dim=None, mult=2, norm=True, groups=1):
        super().__init__()

        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, dim),
            )
            if emb_dim is not None
            else None
        )

        self.ds_conv = nn.Conv1d(
            dim, dim, 7, padding=3, groups=dim, padding_mode="reflect"
        )

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv1d(
                dim, dim_out * mult, 7, 1, 3, padding_mode="reflect", groups=groups
            ),
            nn.SiLU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv1d(
                dim_out * mult, dim_out, 7, 1, 3, padding_mode="reflect", groups=groups
            ),
        )

        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if (self.mlp is not None) and (time_emb is not None):
            condition = self.mlp(time_emb)
            h = h + condition.unsqueeze(-1)

        h = self.net(h)
        return h + self.res_conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        h_dims,
        h_dim_groups,
        convnext_mult,
        wave_stack_depth,
        wave_num_stacks,
        blocks_per_depth,
        attn_heads,
        attn_dim,
        learned_sinu_pos_emb_dim,
    ):
        super().__init__()

        block = partial(ConvNextBlock, mult=convnext_mult, groups=h_dim_groups)

        in_out = list(zip(h_dims[:-1], h_dims[1:]))
        num_layers = len(in_out)

        self.init_conv = nn.Sequential(
            nn.Conv1d(in_dim, h_dims[0], 7, padding=3),
            WaveBlock(h_dims[0], wave_stack_depth, wave_num_stacks),
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
                                    emb_dim=emb_dim,
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
        self.mid_block1 = block(mid_dim, mid_dim, emb_dim=emb_dim)
        self.mid_attn = Residual(
            PreNorm(mid_dim, Attention(mid_dim, heads=attn_heads, dim_head=attn_dim))
        )
        self.mid_block2 = block(mid_dim, mid_dim, emb_dim=emb_dim)

        self.ups = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                block(
                                    dim_out * 2 if i == 0 else dim_in,
                                    dim_in,
                                    emb_dim=emb_dim,
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
