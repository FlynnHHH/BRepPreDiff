from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from brepprediff.data.graph import GraphBatch
from brepprediff.models.downstream import DownstreamEncoder
from brepprediff.task import SEGMENTATION


# Preserve the AdaLN head layout of the archived label-diffusion checkpoints.
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
    """Small MAR-style AdaLN MLP predicting a configured target for one label token."""

    def __init__(
        self,
        in_channels: int,
        condition_dim: int,
        *,
        width: int,
        depth: int,
        dropout: float = 0.0,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels if out_channels is not None else in_channels)
        self.input_projection = nn.Linear(in_channels, width)
        self.condition_projection = nn.Linear(condition_dim, width)
        self.time_embedding = TimestepEmbedder(width)
        self.blocks = nn.ModuleList([AdaLNResidualBlock(width, dropout) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1.0e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 2))
        self.output_projection = nn.Linear(width, self.out_channels)
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


class DiffusionSegmentationModel(DownstreamEncoder):
    """Inference compatibility for archived one-step x_start segmentation heads."""

    task = SEGMENTATION

    def __init__(self, config: dict[str, Any], face_cont_dim: int, edge_cont_dim: int) -> None:
        super().__init__(config, face_cont_dim, edge_cont_dim)
        diffusion_cfg = config['label_diffusion']
        self._validate_sampling(diffusion_cfg)
        hidden_dim = int(config['model']['hidden_dim'])
        self.num_classes = int(config['model']['num_classes'])
        self.timesteps = int(diffusion_cfg.get('train_timesteps', 1000))
        if self.timesteps <= 0:
            raise ValueError('label_diffusion.train_timesteps must be positive.')
        self.diffusion_head = ConditionalDenoisingMLP(
            self.num_classes,
            hidden_dim,
            width=int(diffusion_cfg.get('head_width', hidden_dim)),
            depth=int(diffusion_cfg.get('head_depth', 3)),
            dropout=float(diffusion_cfg.get('head_dropout', 0.0)),
        )

    @staticmethod
    def _validate_sampling(config: dict[str, Any]) -> None:
        if (
            config.get('target_encoding', 'bipolar_one_hot') != 'bipolar_one_hot'
            or config.get('prediction_type', 'epsilon') != 'x_start'
            or config.get('sampling_method', 'ddim') != 'ddim'
            or int(config.get('sampling_steps', 25)) != 1
            or float(config.get('ddim_eta', 0.0)) != 0.0
        ):
            raise ValueError(
                'Archived diffusion inference requires bipolar_one_hot targets, '
                'prediction_type=x_start, sampling_method=ddim, sampling_steps=1, '
                'and ddim_eta=0.0, as configured in ckpt/cadsynth_best.pt.'
            )

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        raise RuntimeError(
            'Archived diffusion heads support evaluation and inference only. '
            'Use predict_segmentation_probabilities to run this checkpoint; '
            'new fine-tuning runs should use an MLP or linear head.'
        )

    def predict_probabilities(self, batch: GraphBatch, config: dict[str, Any]) -> torch.Tensor:
        diffusion_cfg = config['label_diffusion']
        self._validate_sampling(diffusion_cfg)
        condition = self.encode_faces(batch)
        timesteps = torch.full(
            (condition.shape[0],), self.timesteps - 1,
            device=condition.device, dtype=torch.long,
        )
        graph_ptr = batch.graph_ptr.detach().cpu().tolist()
        inference_samples = max(1, int(diffusion_cfg.get('inference_samples', 1)))
        score_temperature = max(float(diffusion_cfg.get('score_temperature', 1.0)), 1.0e-6)
        base_seed = int(diffusion_cfg.get('seed', config.get('seed', 42)))
        score_bias = diffusion_cfg.get('class_score_bias')
        if score_bias is not None:
            if len(score_bias) != self.num_classes:
                raise ValueError('label_diffusion.class_score_bias must contain one value per class.')
            score_bias = condition.new_tensor(score_bias)
        probabilities = condition.new_zeros((condition.shape[0], self.num_classes))
        for sample_index in range(inference_samples):
            noise_chunks = []
            for index, sample_id in enumerate(batch.sample_ids):
                digest = hashlib.sha256(f'{base_seed}:{sample_index}:{sample_id}'.encode('utf-8')).digest()
                seed = int.from_bytes(digest[:8], byteorder='little', signed=False) % (2**63 - 1)
                generator = torch.Generator(device=condition.device).manual_seed(seed)
                noise_chunks.append(torch.randn(
                    (graph_ptr[index + 1] - graph_ptr[index], self.num_classes),
                    dtype=condition.dtype, device=condition.device, generator=generator,
                ))
            noise = torch.cat(noise_chunks, dim=0) * float(diffusion_cfg.get('sampling_temperature', 1.0))
            # The final step of x_start DDIM returns the head output directly;
            # the noise schedule has no effect when sampling_steps is one.
            sampled = self.diffusion_head(noise, timesteps, condition)
            if bool(diffusion_cfg.get('clip_x_start', True)):
                sampled = sampled.clamp(-1.0, 1.0)
            scores = sampled / score_temperature
            if score_bias is not None:
                scores = scores + score_bias
            probabilities.add_(F.softmax(scores, dim=-1))
        return probabilities / inference_samples
