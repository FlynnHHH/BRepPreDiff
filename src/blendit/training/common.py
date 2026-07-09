from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from blendit.config import apply_overrides, feature_dims, load_config, save_config
from blendit.utils import create_run_dir, make_logger, seed_everything


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = ""

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class WeightLoadResult:
    loaded_keys: list[str]
    missing_keys: list[str]
    unexpected_keys: list[str]
    skipped_keys: list[str]


class MetricAverager:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.count = 0

    def update(self, metrics: dict[str, float]) -> None:
        self.count += 1
        for key, value in metrics.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value)

    def compute(
        self,
        *,
        distributed: DistributedContext | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, float]:
        if self.count == 0:
            return {}
        if distributed is not None and distributed.enabled:
            keys = sorted(self.totals)
            reduce_device = device if distributed.backend == "nccl" else torch.device("cpu")
            values = [float(self.count), *(self.totals[key] for key in keys)]
            tensor = torch.tensor(values, dtype=torch.float64, device=reduce_device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            total_count = tensor[0].item()
            if total_count == 0:
                return {}
            return {
                key: float(tensor[index + 1].item() / total_count)
                for index, key in enumerate(keys)
            }
        return {key: value / self.count for key, value in self.totals.items()}


def parse_train_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", action="append", default=[], help="Override config, e.g. train.epochs=5")
    return parser.parse_args()


def load_train_config(args: argparse.Namespace, stage: str) -> dict[str, Any]:
    config = apply_overrides(load_config(args.config), args.override)
    config["train"]["stage"] = stage
    return config


def setup_distributed(config: dict[str, Any]) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    requested = str(config["train"].get("device", "cpu"))
    use_cuda = requested.startswith("cuda") and torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"

    if use_cuda:
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            raise RuntimeError(f"LOCAL_RANK={local_rank} but only {device_count} CUDA devices are visible.")
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return DistributedContext(
        enabled=True,
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
        backend=backend,
    )


def cleanup_distributed(distributed: DistributedContext) -> None:
    if distributed.enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _null_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def prepare_run(
    args: argparse.Namespace,
    stage: str,
    *,
    config: dict[str, Any] | None = None,
    distributed: DistributedContext | None = None,
) -> tuple[dict[str, Any], Path, Any]:
    if config is None:
        config = load_train_config(args, stage)
    else:
        config["train"]["stage"] = stage
    distributed = distributed or DistributedContext()
    seed_everything(int(config.get("seed", 42)))

    if distributed.is_main_process:
        run_dir = create_run_dir(config["run"]["output_dir"], stage, config["run"].get("name"))
        save_config(config, run_dir / "config.yaml")
        run_dir_obj: list[str | None] = [str(run_dir)]
    else:
        run_dir_obj = [None]

    if distributed.enabled:
        dist.broadcast_object_list(run_dir_obj, src=0)
    if run_dir_obj[0] is None:
        raise RuntimeError("Failed to resolve run directory.")

    run_dir = Path(run_dir_obj[0])
    logger = (
        make_logger(run_dir / "logs" / f"{stage}.log")
        if distributed.is_main_process
        else _null_logger(f"blendit.rank{distributed.rank}")
    )
    face_dim, edge_dim = feature_dims(config)
    logger.info(
        "run_dir=%s face_cont_dim=%d edge_cont_dim=%d world_size=%d",
        run_dir,
        face_dim,
        edge_dim,
        distributed.world_size,
    )
    return config, run_dir, logger


def resolve_device(config: dict[str, Any], distributed: DistributedContext | None = None) -> torch.device:
    distributed = distributed or DistributedContext()
    requested = str(config["train"].get("device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if distributed.enabled and requested.startswith("cuda"):
        return torch.device(f"cuda:{distributed.local_rank}")
    return torch.device(requested)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def maybe_wrap_ddp(model: torch.nn.Module, distributed: DistributedContext) -> torch.nn.Module:
    if not distributed.enabled:
        return model
    if distributed.backend == "nccl":
        return DistributedDataParallel(
            model,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
            find_unused_parameters=True,
        )
    return DistributedDataParallel(model, find_unused_parameters=True)


def _normalize_state_dict_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, float] | None = None,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "config": config,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = _normalize_state_dict_keys(checkpoint.get("model", checkpoint))
    unwrap_model(model).load_state_dict(state)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0))


def _checkpoint_model_state(path: str | Path, device: torch.device | str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint does not contain a model state dict: {path}")
    return _normalize_state_dict_keys(state)


def _load_mapped_weights(
    path: str | Path,
    *,
    model: torch.nn.Module,
    device: torch.device | str = "cpu",
    key_mappings: tuple[tuple[str, str], ...],
) -> WeightLoadResult:
    state = _checkpoint_model_state(path, device)
    target_model = unwrap_model(model)
    target_state = target_model.state_dict()
    mapped_state: dict[str, torch.Tensor] = {}
    skipped_keys: list[str] = []

    for source_prefix, target_prefix in key_mappings:
        for key, value in state.items():
            if not key.startswith(source_prefix):
                continue
            target_key = f"{target_prefix}{key.removeprefix(source_prefix)}"
            if target_key not in target_state:
                skipped_keys.append(f"{key} -> {target_key} (missing target)")
                continue
            if target_state[target_key].shape != value.shape:
                skipped_keys.append(
                    f"{key} -> {target_key} "
                    f"(shape {tuple(value.shape)} != {tuple(target_state[target_key].shape)})"
                )
                continue
            mapped_state[target_key] = value

    if not mapped_state:
        mappings = ", ".join(f"{src}* -> {dst}*" for src, dst in key_mappings)
        raise ValueError(f"No compatible weights found in checkpoint {path} for mappings: {mappings}")

    incompatible = target_model.load_state_dict(mapped_state, strict=False)
    return WeightLoadResult(
        loaded_keys=sorted(mapped_state),
        missing_keys=sorted(incompatible.missing_keys),
        unexpected_keys=sorted(incompatible.unexpected_keys),
        skipped_keys=sorted(skipped_keys),
    )


def load_pretrained_encoder(
    path: str | Path,
    *,
    model: torch.nn.Module,
    device: torch.device | str = "cpu",
) -> WeightLoadResult:
    return _load_mapped_weights(
        path,
        model=model,
        device=device,
        key_mappings=(("encoder.", "encoder."),),
    )


def load_pretrain_checkpoint_for_finetune(
    path: str | Path,
    *,
    model: torch.nn.Module,
    device: torch.device | str = "cpu",
) -> WeightLoadResult:
    return _load_mapped_weights(
        path,
        model=model,
        device=device,
        key_mappings=(
            ("encoder.", "encoder."),
            ("coarse_label_head.", "seg_head."),
        ),
    )


def class_weights_from_config(config: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    weights = config["train"].get("class_weights")
    if weights is None:
        return None
    return torch.tensor([float(x) for x in weights], dtype=torch.float32, device=device)


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.5f}" for key, value in sorted(metrics.items()))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in unwrap_model(model).parameters())


def log_config_summary(logger: Any, config: dict[str, Any]) -> None:
    data_cfg = config["data"]
    train_cfg = config["train"]
    logger.info(
        "config data.cache_only=%s data.cache_dir=%s data.train_split=%s data.val_split=%s",
        data_cfg.get("cache_only", False),
        data_cfg.get("cache_dir"),
        data_cfg.get("train_split"),
        data_cfg.get("val_split"),
    )
    logger.info(
        "config train.epochs=%s train.batch_size=%s train.lr=%s train.weight_decay=%s "
        "train.resume=%s train.num_workers=%s",
        train_cfg.get("epochs"),
        train_cfg.get("batch_size"),
        train_cfg.get("lr"),
        train_cfg.get("weight_decay"),
        train_cfg.get("resume"),
        data_cfg.get("num_workers", 0),
    )


def log_dataloader_summary(logger: Any, name: str, dataloader: Any, distributed: DistributedContext) -> None:
    dataset = dataloader.dataset
    sampler = getattr(dataloader, "sampler", None)
    logger.info(
        "dataloader %s ready: samples=%d batches_per_rank=%d batch_size_per_rank=%s "
        "sampler=%s distributed=%s world_size=%d",
        name,
        len(dataset),
        len(dataloader),
        getattr(dataloader, "batch_size", None),
        type(sampler).__name__ if sampler is not None else "None",
        distributed.enabled,
        distributed.world_size,
    )


def log_checkpoint_saved(logger: Any, path: str | Path, epoch: int) -> None:
    logger.info("checkpoint saved path=%s epoch=%d", path, epoch)
