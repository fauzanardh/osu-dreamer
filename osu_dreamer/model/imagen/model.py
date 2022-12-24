from typing import Tuple
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, reduce

from osu_dreamer.model.imagen import GaussianDiffusionContinuousTimes
from osu_dreamer.model.imagen import UNet
from osu_dreamer.data import A_DIM
from osu_dreamer.signal import MAP_SIGNAL_DIM as X_DIM


VALID_PAD = 1024


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


class Model(nn.Module):
    def __init__(
        self,
        h_dim: int = 128,
        h_dim_mult: Tuple[int, int, int, int] = (1, 2, 4, 4),
        cond_dim=None,
        num_time_tokens: int = 2,
        learned_sinu_pos_emb_dim: int = 16,
        layer_attns: Tuple[bool, bool, bool, bool] = (False, True, True, True),
        layer_cross_attns: Tuple[bool, bool, bool, bool] = (False, False, False, False),
        num_resnet_blocks: int = 2,
        resnet_groups: int = 8,
        scale_skip_connection: bool = True,
        use_global_context_attn: bool = True,
        attn_heads: int = 8,
        attn_dim_head: int = 64,
        attn_depth: int = 1,
        ff_mult: int = 2,
        timesteps: int = 1024,
        loss_type: str = "l2",
        noise_schedule: str = "cosine",
        pred_objective: str = "noise",
        dynamic_thresholding: bool = True,
        dynamic_thresholding_percentile: float = 0.95,
    ):
        super().__init__()

        # loss
        if loss_type == "l1":
            self.loss_fn = F.l1_loss
        elif loss_type == "l2":
            self.loss_fn = F.mse_loss
        elif loss_type == "huber":
            self.loss_fn = F.smooth_l1_loss
        else:
            raise ValueError(f"Invalid loss type {loss_type}")

        self.noise_scheduler = GaussianDiffusionContinuousTimes(
            noise_schedule=noise_schedule,
            timesteps=timesteps,
        )

        self.unet = UNet(
            dim_in=A_DIM + X_DIM,
            dim_out=X_DIM,
            cond_dim=cond_dim,
            cond_on_map_difficulty=False,
            num_time_tokens=num_time_tokens,
            learned_sinu_pos_emb_dim=learned_sinu_pos_emb_dim,
            h_dim=h_dim,
            h_dim_mult=h_dim_mult,
            layer_attns=layer_attns,
            layer_cross_attns=layer_cross_attns,
            num_resnet_blocks=num_resnet_blocks,
            resnet_groups=resnet_groups,
            scale_skip_connection=scale_skip_connection,
            use_global_context_attn=use_global_context_attn,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            attn_depth=attn_depth,
            ff_mult=ff_mult,
        )
        self.unet_depth = len(h_dim_mult)

        self.pred_objective = pred_objective

        self.dynamic_thresholding = dynamic_thresholding
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile

        # one temp parameter for keeping track of device
        self.register_buffer("_temp", torch.tensor([0.0]), persistent=False)

    @property
    def device(self):
        return self._temp.device

    def inference_pad(self, x):
        x = F.pad(x, (VALID_PAD, VALID_PAD), mode="replicate")
        pad = (1 + x.size(-1) // 2**self.unet_depth) * 2**self.unet_depth - x.size(
            -1
        )
        x = F.pad(x, (0, pad), mode="replicate")
        return x, (..., slice(VALID_PAD, -(VALID_PAD + pad)))

    def p_mean_variance(self, x, a, t, *, t_next=None, model_output=None):
        if model_output is None:
            model_output = self.unet(x, a, t)

        pred = model_output

        if self.pred_objective == "noise":
            x_start = self.noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)
        elif self.pred_objective == "x_start":
            x_start = pred
        elif self.pred_objective == "v":
            x_start = self.noise_scheduler.predict_start_from_v(x, t=t, v=pred)
        else:
            raise ValueError(f"Invalid pred objective {self.pred_objective}")

        if self.dynamic_thresholding:
            s = torch.quantile(
                rearrange(x_start, "b ... -> b (...)").abs(),
                self.dynamic_thresholding_percentile,
                dim=-1,
            )
            s.clamp_(min=1.0)
            s = right_pad_dims_to(x_start, s)
            x_start = x_start.clamp(-s, s) / s
        else:
            x_start.clamp_(min=-1.0, max=1.0)

        mean_and_variance = self.noise_scheduler.q_posterior(
            x_start=x_start,
            x_t=x,
            t=t,
            t_next=t_next,
        )

        return mean_and_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, a, t, *, t_next=None):
        (model_mean, _, model_log_variance), x_start = self.p_mean_variance(
            x=x, a=a, t=t, t_next=t_next
        )

        noise = torch.randn_like(x)
        # no noise when t == 0
        is_last_sampling_step = t_next == 0
        nonzero_mask = 1 - is_last_sampling_step.float()
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(self, a, shape, resample_times):
        device = self.device

        batch = shape[0]
        x = torch.randn(shape, device=device)

        timesteps = self.noise_scheduler.get_sampling_timesteps(batch, device=device)

        for times, times_next in tqdm(
            timesteps,
            desc="Sampling",
            total=len(timesteps),
        ):
            x, _ = self.p_sample(
                x=x,
                a=a,
                t=times,
                t_next=times_next,
            )

        x.clamp_(min=-1.0, max=1.0)
        return x

    @torch.no_grad()
    def sample(self, a, resample_times=35):
        batch = a.shape[0]
        length = a.shape[-1]
        shape = (batch, X_DIM, length)

        x = self.p_sample_loop(
            a=a,
            shape=shape,
            resample_times=resample_times,
        )

        return x

    def p_losses(
        self,
        x_start,
        a,
        times,
        *,
        noise=None,
    ):
        x_start, sl = self.inference_pad(x_start)
        a, _ = self.inference_pad(a)

        if noise is None:
            noise = torch.randn_like(x_start)

        # get x_t
        x_noisy, log_snr, alpha, sigma = self.noise_scheduler.q_sample(
            x_start=x_start,
            t=times,
            noise=noise,
        )

        # time condition
        noise_cond = self.noise_scheduler.get_condition(times)

        pred = self.unet(x_noisy, a, noise_cond)

        if self.pred_objective == "noise":
            target = noise
        elif self.pred_objective == "x_start":
            target = x_start
        elif self.pred_objective == "v":
            target = alpha * noise - sigma * x_start
        else:
            raise ValueError(f"Invalid pred objective {self.pred_objective}")

        # unpad
        pred = pred[sl]
        target = target[sl]

        # lossses
        losses = self.loss_fn(pred, target, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")

        return losses.mean()

    def forward(self, x, a):
        b = x.shape[0]
        times = self.noise_scheduler.sample_random_times(b, device=self.device)

        return self.p_losses(
            x_start=x,
            a=a,
            times=times,
        )
