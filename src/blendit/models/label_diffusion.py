from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


def bipolar_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Encode class ids as vertices in {-1, +1}^C."""
    return F.one_hot(labels, num_classes=num_classes).to(dtype=torch.float32).mul_(2.0).sub_(1.0)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    if timesteps <= 0:
        raise ValueError(f"timesteps must be positive, got {timesteps}")
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / timesteps + s) / (1.0 + s)) * math.pi * 0.5).pow(2)
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1.0e-8, 0.999).to(dtype=torch.float32)


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    if timesteps <= 0:
        raise ValueError(f"timesteps must be positive, got {timesteps}")
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def _extract(values: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    extracted = values.gather(0, timesteps)
    return extracted.reshape((timesteps.shape[0],) + (1,) * (reference.ndim - 1))


class LabelDiffusionSchedule(nn.Module):
    """Forward label noising and deterministic/stochastic DDIM sampling."""

    def __init__(
        self,
        timesteps: int,
        *,
        noise_schedule: str = "cosine",
        beta_start: float = 1.0e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        if noise_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif noise_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unsupported label diffusion noise schedule: {noise_schedule!r}")

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.timesteps = int(timesteps)
        self.noise_schedule = noise_schedule
        self.register_buffer("alpha_bar", alpha_bar, persistent=False)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar), persistent=False)
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar), persistent=False)

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_start)
        scale_clean = _extract(self.sqrt_alpha_bar, timesteps, x_start)
        scale_noise = _extract(self.sqrt_one_minus_alpha_bar, timesteps, x_start)
        return scale_clean * x_start + scale_noise * noise, noise

    def predict_x_start(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        scale_clean = _extract(self.sqrt_alpha_bar, timesteps, x_t)
        scale_noise = _extract(self.sqrt_one_minus_alpha_bar, timesteps, x_t)
        return (x_t - scale_noise * epsilon) / scale_clean.clamp_min(1.0e-8)

    def sampling_timesteps(self, steps: int, device: torch.device) -> torch.Tensor:
        if steps <= 0:
            raise ValueError(f"sampling steps must be positive, got {steps}")
        steps = min(int(steps), self.timesteps)
        return torch.linspace(self.timesteps - 1, 0, steps, device=device).round().long()

    def sample(
        self,
        denoiser: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        condition: torch.Tensor,
        *,
        initial_noise: torch.Tensor | None = None,
        steps: int,
        eta: float = 0.0,
        temperature: float = 1.0,
        clip_x_start: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if initial_noise is None:
            x_t = torch.randn(
                (condition.shape[0], denoiser.in_channels),
                dtype=condition.dtype,
                device=condition.device,
                generator=generator,
            )
        else:
            x_t = initial_noise.to(dtype=condition.dtype, device=condition.device)
        x_t = x_t * float(temperature)

        sampling_times = self.sampling_timesteps(steps, condition.device)
        for index, timestep in enumerate(sampling_times):
            t = torch.full((condition.shape[0],), int(timestep.item()), device=condition.device, dtype=torch.long)
            epsilon = denoiser(x_t, t, condition)
            x_start = self.predict_x_start(x_t, t, epsilon)
            if clip_x_start:
                x_start = x_start.clamp(-1.0, 1.0)

            if index + 1 == sampling_times.numel():
                x_t = x_start
                continue

            previous_timestep = sampling_times[index + 1]
            alpha_t = self.alpha_bar[timestep]
            alpha_previous = self.alpha_bar[previous_timestep]
            sigma = float(eta) * torch.sqrt(
                ((1.0 - alpha_previous) / (1.0 - alpha_t)).clamp_min(0.0)
                * (1.0 - alpha_t / alpha_previous).clamp_min(0.0)
            )
            direction = torch.sqrt((1.0 - alpha_previous - sigma.square()).clamp_min(0.0)) * epsilon
            if float(eta) > 0.0:
                step_noise = torch.randn(
                    x_t.shape,
                    dtype=x_t.dtype,
                    device=x_t.device,
                    generator=generator,
                )
            else:
                step_noise = torch.zeros_like(x_t)
            x_t = torch.sqrt(alpha_previous) * x_start + direction + sigma * step_noise
        return x_t


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.frequency_dim = int(frequency_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def sinusoidal_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
            / max(half, 1)
        )
        args = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal_embedding(timesteps, self.frequency_dim))


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class AdaLNResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=1.0e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 3))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        return x + gate * self.mlp(_modulate(self.norm(x), shift, scale))


class ConditionalDenoisingMLP(nn.Module):
    """Small MAR-style AdaLN MLP predicting epsilon for one label token."""

    def __init__(
        self,
        in_channels: int,
        condition_dim: int,
        *,
        width: int,
        depth: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.input_projection = nn.Linear(in_channels, width)
        self.condition_projection = nn.Linear(condition_dim, width)
        self.time_embedding = TimestepEmbedder(width)
        self.blocks = nn.ModuleList([AdaLNResidualBlock(width, dropout) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1.0e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 2))
        self.output_projection = nn.Linear(width, in_channels)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        nn.init.normal_(self.time_embedding.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedding.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x_t)
        adaptive_condition = self.time_embedding(timesteps) + self.condition_projection(condition)
        for block in self.blocks:
            hidden = block(hidden, adaptive_condition)
        shift, scale = self.final_modulation(adaptive_condition).chunk(2, dim=-1)
        hidden = _modulate(self.final_norm(hidden), shift, scale)
        return self.output_projection(hidden)
