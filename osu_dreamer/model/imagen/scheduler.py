import math
from functools import partial

import torch
from torch.special import expm1
from torch import nn

from einops import repeat


def log(t, eps: float = 1e-12):
    return torch.log(t.clamp(min=eps))


def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


@torch.jit.script
def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t**2)))


@torch.jit.script
def alpha_cosine_log_snr(t, s: float = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps=1e-5)


class GaussianDiffusionContinuousTimes(nn.Module):
    def __init__(self, *, noise_schedule, timesteps=1024):
        super().__init__()

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f"Unknown noise schedule {noise_schedule}")

        self.num_timesteps = timesteps

    def get_times(self, batch_size, noise_level, *, device):
        return torch.full(
            (batch_size,), noise_level, device=device, dtype=torch.float32
        )

    def sample_random_times(self, batch_size, *, device):
        return torch.zeros((batch_size,), device=device).float().uniform_(0, 1)

    def get_condition(self, times):
        if times is None:
            return times
        else:
            return self.log_snr(times)

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1.0, 0.0, self.num_timesteps + 1, device=device)
        times = repeat(times, "t -> b t", b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    def q_posterior(self, x_start, x_t, t, *, t_next=None):
        if t_next is None:
            t_next = (t - 1.0 / self.num_timesteps).clamp(min=0.0)

        """ https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material """
        log_snr = self.log_snr(t)
        log_snr_next = self.log_snr(t_next)
        log_snr, log_snr_next = map(
            partial(right_pad_dims_to, x_t), (log_snr, log_snr_next)
        )

        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

        # c - as defined near eq 33
        c = -expm1(log_snr - log_snr_next)
        posterior_mean = alpha_next * (x_t * (1 - c) / alpha + c * x_start)

        # following (eq. 33)
        posterior_variance = (sigma_next**2) * c
        posterior_log_variance_clipped = log(posterior_variance, eps=1e-20)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        dtype = x_start.dtype

        if isinstance(t, float):
            batch = x_start.shape[0]
            t = torch.full((batch,), t, device=x_start.device, dtype=dtype)

        if noise is None:
            noise = torch.randn_like(x_start)
        log_snr = self.log_snr(t).type(dtype)
        log_snr_padded_dim = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr_padded_dim)

        return alpha * x_start + sigma * noise, log_snr, alpha, sigma

    def q_sample_from_to(self, x_from, from_t, to_t, noise=None):
        shape, device, dtype = x_from.shape, x_from.device, x_from.dtype
        batch = shape[0]

        if isinstance(from_t, float):
            from_t = torch.full((batch,), from_t, device=device, dtype=dtype)

        if isinstance(to_t, float):
            to_t = torch.full((batch,), to_t, device=device, dtype=dtype)

        if noise is None:
            noise = torch.randn_like(x_from)

        log_snr = self.log_snr(from_t)
        log_snr_padded_dim = right_pad_dims_to(x_from, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr_padded_dim)

        log_snr_to = self.log_snr(to_t)
        log_snr_padded_dim_to = right_pad_dims_to(x_from, log_snr_to)
        alpha_to, sigma_to = log_snr_to_alpha_sigma(log_snr_padded_dim_to)

        return (
            x_from * (alpha_to / alpha)
            + noise * (sigma_to * alpha - sigma * alpha_to) / alpha
        )

    def predict_start_from_v(self, x_t, t, v):
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return alpha * x_t - sigma * v

    def predict_start_from_noise(self, x_t, t, noise):
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return (x_t - sigma * noise) / alpha.clamp(min=1e-8)
