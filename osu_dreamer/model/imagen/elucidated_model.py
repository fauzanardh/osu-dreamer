from math import sqrt
from tqdm.auto import tqdm
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, reduce

from osu_dreamer.model.imagen import UNet
from osu_dreamer.signal import X_DIM


VALID_PAD = 1024
A_DIM = 40


class Hparams(NamedTuple):
    num_sample_steps: int
    sigma_min: float
    sigma_max: float
    sigma_data: float
    rho: float
    P_mean: float
    P_std: float
    S_churn: float
    S_tmin: float
    S_tmax: float
    S_noise: float


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


class ElucidatedModel(nn.Module):
    def __init__(
        self,
        h_dim=128,
        h_dim_mult=(1, 2, 3, 4),
        cond_dim=None,
        num_time_tokens=2,
        learned_sinu_pos_emb_dim=16,
        layer_attns=(False, True, True, True),
        layer_cross_attns=(False, True, True, True),
        num_resnet_blocks=2,
        resnet_groups=8,
        scale_skip_connection=True,
        attn_heads=8,
        attn_dim_head=64,
        attn_depth=1,
        ff_mult=2,
        loss_type="l2",
        dynamic_thresholding=True,
        dynamic_thresholding_percentile=0.95,
        num_sample_steps=35,
        sigma_min=0.002,
        sigma_max=80,
        sigma_data=0.5,
        rho=7,
        P_mean=-1.2,
        P_std=1.2,
        S_churn=80,
        S_tmin=0.05,
        S_tmax=50,
        S_noise=1.003,
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
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            attn_depth=attn_depth,
            ff_mult=ff_mult,
        )
        self.unet_depth = len(h_dim_mult)

        hparams = [
            num_sample_steps,
            sigma_min,
            sigma_max,
            sigma_data,
            rho,
            P_mean,
            P_std,
            S_churn,
            S_tmin,
            S_tmax,
            S_noise,
        ]

        self.hparams = Hparams(*hparams)

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

    # dynamic thresholding
    def threshold_x_start(self, x_start):
        if self.dynamic_thresholding:
            s = torch.quantile(
                rearrange(x_start, "b ... -> b (...)").abs(),
                self.dynamic_thresholding_percentile,
                dim=-1,
            )
            s.clamp_(min=1.0)
            s = right_pad_dims_to(x_start, s)
            return x_start.clamp(-s, s) / s
        else:
            return x_start.clamp(-1.0, 1.0)

    # preconditioning params
    def c_skip(self, sigma_data, sigma):
        return (sigma_data**2) / (sigma**2 + sigma_data**2)

    def c_out(self, sigma_data, sigma):
        return sigma * sigma_data * (sigma_data**2 + sigma**2) ** -0.5

    def c_in(self, sigma_data, sigma):
        return 1 * (sigma**2 + sigma_data**2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output
    def preconditioned_network_forward(
        self,
        noised_map,
        a,
        sigma,
        *,
        sigma_data,
        clamp=False,
        **kwargs,
    ):
        b, device = noised_map.shape[0], noised_map.device

        if isinstance(sigma, float):
            sigma = torch.full((b,), sigma, device=device)
        padded_sigma = rearrange(sigma, "b -> b 1 1")

        net_out = self.unet(
            self.c_in(sigma_data, padded_sigma) * noised_map,
            a,
            self.c_noise(sigma),
            **kwargs,
        )

        out = (
            self.c_skip(sigma_data, padded_sigma) * noised_map
            + self.c_out(sigma_data, padded_sigma) * net_out
        )

        if clamp:
            return self.threshold_x_start(out)
        else:
            return out

    # sample schedule
    def sample_schedule(self, num_sample_steps, rho, sigma_min, sigma_max):
        N = num_sample_steps
        inv_rho = 1 / rho

        steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (
            sigma_max**inv_rho
            + steps / (N - 1) * (sigma_min**inv_rho - sigma_max**inv_rho)
        ) ** rho

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    def sample(
        self,
        a,
        *,
        clamp=True,
        sigma_min=None,
        sigma_max=None,
        **kwargs,
    ):
        a, sl = self.inference_pad(a)
        b = a.shape[0]
        n = a.shape[-1]
        shape = (b, X_DIM, n)

        hp = self.hparams

        if sigma_min is None:
            sigma_min = hp.sigma_min

        if sigma_max is None:
            sigma_max = hp.sigma_max

        sigmas = self.sample_schedule(hp.num_sample_steps, hp.rho, sigma_min, sigma_max)

        gammas = torch.where(
            (sigmas >= hp.S_tmin) & (sigmas <= hp.S_tmax),
            min(hp.S_churn / hp.num_sample_steps, sqrt(2) - 1),
            0.0,
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # maps is noise at the beginning
        init_sigma = sigmas[0]

        maps = init_sigma * torch.randn(shape, device=self.device)

        total_steps = len(sigmas_and_gammas)

        for sigma, sigma_next, gamma in tqdm(
            sigmas_and_gammas,
            total=total_steps,
            desc="Sampling time steps",
        ):
            sigma_hat = sigma + gamma * sigma
            if gamma > 0:
                eps = hp.S_noise * torch.randn(
                    shape,
                    device=self.device,
                )
                added_noise = sqrt(sigma_hat**2 - sigma**2) * eps
            else:
                added_noise = 0.0

            maps_hat = maps + added_noise

            if len(sigma_hat.shape) == 0:
                sigma_hat = sigma_hat.unsqueeze(0)

            model_output = self.preconditioned_network_forward(
                maps_hat,
                a,
                sigma_hat,
                sigma_data=hp.sigma_data,
                clamp=clamp,
                **kwargs,
            )

            denoised_over_sigma = (maps_hat - model_output) / sigma_hat

            maps_next = maps_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            if sigma_next != 0:
                if len(sigma_next.shape) == 0:
                    sigma_next = sigma_next.unsqueeze(0)

                model_output_next = self.preconditioned_network_forward(
                    maps_next,
                    a,
                    sigma_next,
                    sigma_data=hp.sigma_data,
                    clamp=clamp,
                    **kwargs,
                )

                denoised_prime_over_sigma = (maps_next - model_output_next) / sigma_next
                maps_next = maps_hat + 0.5 * (sigma_next - sigma_hat) * (
                    denoised_over_sigma + denoised_prime_over_sigma
                )

            maps = maps_next

        maps = maps.clamp(-1.0, 1.0)
        return maps[sl]

    def loss_weight(self, sigma_data, sigma):
        return (sigma**2 + sigma_data**2) * (sigma * sigma_data) ** -2

    def noise_distribution(self, P_mean, P_std, b):
        return (P_mean + P_std * torch.randn((b,), device=self.device)).exp()

    def forward(
        self,
        maps,
        a,
    ):
        b = maps.shape[0]
        hp = self.hparams

        # get the sigmas
        sigmas = self.noise_distribution(hp.P_mean, hp.P_std, b)
        padded_sigmas = rearrange(sigmas, "b -> b 1 1")

        # noise
        noise = torch.randn_like(maps, device=self.device)
        noised_maps = maps + padded_sigmas * noise

        # get prediction
        denoised_maps = self.preconditioned_network_forward(
            noised_maps,
            a,
            sigmas,
            sigma_data=hp.sigma_data,
        )

        # losses
        losses = self.loss_fn(denoised_maps, maps, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")

        # loss weighting
        losses = losses * self.loss_weight(hp.sigma_data, sigmas)

        return losses.mean()
