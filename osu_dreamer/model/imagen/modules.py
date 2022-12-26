import math
from functools import partial

import torch
from torch import nn

from xformers.ops import memory_efficient_attention

from einops import rearrange
from einops.layers.torch import Rearrange


class Always(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, *args, **kwargs):
        return self.value


class Parallel(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        outputs = [fn(x) for fn in self.fns]
        return sum(outputs)


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


def FeedForward(dim, mult=2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias=False),
        nn.GELU(),
        LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias=False),
    )


def Downsample(dim, dim_out=None):
    dim_out = dim if dim_out is None else dim_out
    return nn.Conv1d(dim, dim_out, 4, 2, 1, padding_mode='reflect')


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
        context_dim=None,
    ):
        super().__init__()
        self.scale = dim_head**-0.5

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_context = (
            nn.Sequential(
                nn.LayerNorm(context_dim),
                nn.Linear(context_dim, inner_dim * 2),
            )
            if context_dim is not None
            else None
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim),
        )

    def forward(self, x, context=None):
        b = x.shape[0]

        x = self.norm(x)

        q = rearrange(self.to_q(x), "b n (h d) -> b n h d", h=self.heads)
        kv = rearrange(
            self.to_kv(x), "b n (two h d) -> b n two h d", two=2, h=self.heads
        )

        # add conditioning, if present
        if context is not None:
            ckv = rearrange(
                self.to_context(context),
                "b n (two h d) -> b n two h d",
                two=2,
                h=self.heads,
            )
            kv = torch.cat((ckv, kv), dim=-4)

        k, v = kv.unbind(dim=-3)

        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
        out = memory_efficient_attention(q, k, v, scale=self.scale)

        out = rearrange(out, "b n h d -> b n (h d)", b=b, h=self.heads)
        return self.to_out(out.to(x.dtype))


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        norm_context=False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5

        self.heads = heads
        inner_dim = dim_head * heads

        if context_dim is None:
            context_dim = dim

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim),
        )

    def forward(self, x, context):
        b = x.shape[0]

        x = self.norm(x)
        context = self.norm_context(context)

        q = rearrange(self.to_q(x), "b n (h d) -> b n h d", h=self.heads)
        kv = rearrange(
            self.to_kv(context), "b n (two h d) -> b n two h d", two=2, h=self.heads
        )

        k, v = kv.unbind(dim=-3)

        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
        out = memory_efficient_attention(q, k, v, scale=self.scale)

        out = rearrange(out, "b n h d -> b n (h d)", b=b, h=self.heads)
        return self.to_out(out.to(x.dtype))


class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups=8,
        norm=True,
    ):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, dim) if norm else nn.Identity()
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
        cond_dim=None,
        time_cond_dim=None,
        groups=8,
        use_flash_attention=True,
        **attn_kwargs,
    ):
        super().__init__()

        self.time_mlp = None
        if time_cond_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2),
            )

        self.cross_attn = None
        if cond_dim is not None:
            self.cross_attn = CrossAttention(
                dim_out, context_dim=cond_dim, **attn_kwargs
            )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        self.res_conv = nn.Conv1d(dim, dim_out, 7, padding=3, padding_mode="reflect") if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, cond=None):
        scale_shift = None
        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x)

        if self.cross_attn is not None:
            h = rearrange(h, "b c n -> b n c")
            h = self.cross_attn(h, context=cond) + h
            h = rearrange(h, "b n c -> b c n")

        h = self.block2(h, scale_shift=scale_shift)

        return h + self.res_conv(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth=1,
        heads=8,
        dim_head=32,
        ff_mult=2,
        context_dim=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            heads=heads,
                            dim_head=dim_head,
                            context_dim=context_dim,
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x, context=None):
        x = rearrange(x, "b c n -> b n c")

        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x

        return rearrange(x, "b n c -> b c n")


class UNet(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        cond_dim=None,
        cond_on_map_difficulty=False,  # will be implemented later for conditional models
        num_time_tokens=2,
        learned_sinu_pos_emb_dim=16,
        h_dim=128,
        h_dim_mult=(1, 2, 4, 4),
        layer_attns=(False, True, True, True),
        layer_cross_attns=(False, True, True, True),
        num_resnet_blocks=2,
        resnet_groups=8,
        scale_skip_connection=True,
        attn_heads=8,
        attn_dim_head=64,
        attn_depth=1,
        ff_mult=2,
    ):
        super().__init__()
        h_dims = [h_dim, *map(lambda m: h_dim * m, h_dim_mult)]
        in_out = list(zip(h_dims[:-1], h_dims[1:]))

        # time conditioning
        cond_dim = h_dim if cond_dim is None else cond_dim
        time_cond_dim = h_dim * 4

        # embedding time for log(snr) noise from continuous version
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1

        self.to_time_hiddens = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(sinu_pos_emb_input_dim, time_cond_dim),
            nn.SiLU(),
        )

        self.to_time_cond = nn.Linear(time_cond_dim, time_cond_dim)

        # project to time tokens as well as time hiddens
        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange("b (r d) -> b r d", r=num_time_tokens),
        )

        self.norm_cond = nn.LayerNorm(cond_dim)

        # attention related params
        attn_kwargs = dict(
            heads=attn_heads,
            dim_head=attn_dim_head,
        )

        # resnet
        resnet_klass = partial(ResnetBlock, groups=resnet_groups, **attn_kwargs)

        self.init_conv = nn.Conv1d(dim_in, h_dims[0], 7, padding=3)

        # scale skip connection
        self.skip_connect_scale = 1.0 if not scale_skip_connection else (2**-0.5)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolution = len(in_out)

        layer_params = [
            layer_attns,
            layer_cross_attns,
        ]
        reversed_layer_params = list(map(reversed, layer_params))

        # skip connections
        skip_connection_dims = []

        # downsampling layers
        for i, (
            (h_dim_in, h_dim_out),
            layer_attn,
            layer_cross_attn,
        ) in enumerate(zip(in_out, *layer_params)):
            is_last = i >= (num_resolution - 1)

            layer_cond_dim = cond_dim if layer_cross_attn else None

            if layer_attn:
                transformer_block = TransformerBlock
            else:
                transformer_block = nn.Identity

            current_dim = h_dim_in
            skip_connection_dims.append(current_dim)

            post_downsample = (
                Downsample(current_dim, h_dim_out)
                if not is_last
                else Parallel(
                    nn.Conv1d(current_dim, h_dim_out, 7, padding=3),
                    nn.Conv1d(h_dim_in, h_dim_out, 1),
                )
            )

            self.downs.append(
                nn.ModuleList(
                    [
                        resnet_klass(
                            current_dim,
                            current_dim,
                            cond_dim=layer_cond_dim,
                            time_cond_dim=time_cond_dim,
                            groups=resnet_groups,
                        ),
                        nn.ModuleList(
                            [
                                ResnetBlock(
                                    current_dim,
                                    current_dim,
                                    time_cond_dim=time_cond_dim,
                                    groups=resnet_groups,
                                )
                                for _ in range(num_resnet_blocks)
                            ]
                        ),
                        transformer_block(
                            dim=current_dim,
                            depth=attn_depth,
                            ff_mult=ff_mult,
                            context_dim=cond_dim,
                            **attn_kwargs,
                        ),
                        post_downsample,
                    ]
                )
            )

        # middle layers
        mid_dim = h_dims[-1]
        self.mid_block1 = ResnetBlock(
            mid_dim,
            mid_dim,
            cond_dim=cond_dim,
            time_cond_dim=time_cond_dim,
            groups=resnet_groups,
        )
        self.mid_attn = TransformerBlock(
            mid_dim,
            depth=attn_depth,
        )
        self.mid_block2 = ResnetBlock(
            mid_dim,
            mid_dim,
            cond_dim=cond_dim,
            time_cond_dim=time_cond_dim,
            groups=resnet_groups,
        )

        # upsampling layers
        for i, (
            (h_dim_in, h_dim_out),
            layer_attn,
            layer_cross_attn,
        ) in enumerate(zip(reversed(in_out), *reversed_layer_params)):
            is_last = i == (len(in_out) - 1)

            layer_cond_dim = cond_dim if layer_cross_attn else None

            if layer_attn:
                transformer_block = TransformerBlock
            else:
                transformer_block = nn.Identity

            current_dim = h_dim_out
            skip_connection_dim = skip_connection_dims.pop()

            post_upsample = (
                Upsample(current_dim, h_dim_in) if not is_last else nn.Identity()
            )

            self.ups.append(
                nn.ModuleList(
                    [
                        resnet_klass(
                            current_dim + skip_connection_dim,
                            current_dim,
                            cond_dim=layer_cond_dim,
                            time_cond_dim=time_cond_dim,
                            groups=resnet_groups,
                        ),
                        nn.ModuleList(
                            [
                                ResnetBlock(
                                    current_dim + skip_connection_dim,
                                    current_dim,
                                    time_cond_dim=time_cond_dim,
                                    groups=resnet_groups,
                                )
                                for _ in range(num_resnet_blocks)
                            ]
                        ),
                        transformer_block(
                            dim=current_dim,
                            depth=attn_depth,
                            ff_mult=ff_mult,
                            context_dim=cond_dim,
                            **attn_kwargs,
                        ),
                        post_upsample,
                    ]
                )
            )

        # final layers
        self.final_resnet = ResnetBlock(
            h_dim,
            h_dim,
            time_cond_dim=time_cond_dim,
            groups=resnet_groups,
        )
        self.final_conv = nn.Conv1d(h_dim, dim_out, 1)

    def forward(self, x, a, ts):
        x = torch.cat((x, a), dim=1)

        x = self.init_conv(x)

        # time conditioning
        time_hiddens = self.to_time_hiddens(ts)

        # derive time tokens
        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)

        # conditioning tokens, currently only time tokens
        c = time_tokens

        # normalize conditioning tokens
        c = self.norm_cond(c)

        # skip connections
        hiddens = []

        # downsampling layers
        for (
            init_block,
            resnet_blocks,
            attn_block,
            post_downsample,
        ) in self.downs:
            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = resnet_block(x, t)
                hiddens.append(x)

            x = attn_block(x, c)
            hiddens.append(x)

            x = post_downsample(x)

        # middle layers
        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        def add_skip_connection(x):
            return torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim=1)

        # upsampling layers
        for (
            init_block,
            resnet_blocks,
            attn_block,
            post_upsample,
        ) in self.ups:
            x = add_skip_connection(x)
            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = add_skip_connection(x)
                x = resnet_block(x, t)

            x = attn_block(x, c)

            x = post_upsample(x)

        # final layers
        x = self.final_resnet(x, t)

        return self.final_conv(x)
