from __future__ import annotations

import argparse
from datetime import timedelta
import logging
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from blendit.config import feature_dims, load_experiment_config, save_config
from blendit.utils import create_run_dir, make_logger, seed_everything


DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS = 120.0


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


@dataclass(frozen=True)
class EncoderFreezeResult:
    mode: str
    frozen_layers: int
    trainable_parameters: int
    frozen_parameters: int


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
    parser.add_argument(
        "--data-config",
        default=None,
        help="Prepare-data YAML. Defaults to data_config in the training YAML.",
    )
    parser.add_argument("--override", action="append", default=[], help="Override config, e.g. train.epochs=5")
    return parser.parse_args()


def load_train_config(args: argparse.Namespace, stage: str) -> dict[str, Any]:
    config = load_experiment_config(
        args.config,
        getattr(args, "data_config", None),
        args.override,
    )
    config["train"]["stage"] = stage
    return config


def distributed_timeout(config: dict[str, Any]) -> timedelta:
    seconds = float(
        config.get("train", {}).get(
            "distributed_timeout_seconds",
            DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS,
        )
    )
    if seconds <= 0:
        raise ValueError(f"train.distributed_timeout_seconds must be positive, got {seconds}")
    return timedelta(seconds=seconds)


def setup_distributed(config: dict[str, Any]) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    requested = str(config["train"].get("device", "cpu"))
    use_cuda = requested.startswith("cuda") and torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"

    if backend == "nccl" and not any(
        name in os.environ
        for name in ("NCCL_ASYNC_ERROR_HANDLING", "NCCL_BLOCKING_WAIT")
    ):
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

    if use_cuda:
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            raise RuntimeError(f"LOCAL_RANK={local_rank} but only {device_count} CUDA devices are visible.")
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=distributed_timeout(config))

    return DistributedContext(
        enabled=True,
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
        backend=backend,
    )


def prepare_training_data(
    config: dict[str, Any],
    distributed: DistributedContext | None = None,
) -> dict[str, Any]:
    distributed = distributed or DistributedContext()
    prepare_on_start = bool(config.get("data", {}).get("prepare_on_start", False))
    if not prepare_on_start:
        if distributed.is_main_process:
            print("data preparation skipped: data.prepare_on_start=false")
        return config

    from blendit.data.load_data import prepare_data

    if not distributed.enabled:
        result = prepare_data(config)
        print(
            "data preparation complete: "
            f"generated_splits={len(result.generated_splits)} "
            f"built_cache_files={result.built_cache_files} "
            f"removed_invalid_caches={len(result.removed_invalid_caches)}"
        )
        return result.config

    token: list[str | None] = [uuid.uuid4().hex if distributed.is_main_process else None]
    dist.broadcast_object_list(token, src=0)
    if token[0] is None:
        raise RuntimeError("Failed to establish distributed data preparation coordination.")

    cache_dir = Path(config["data"]["cache_dir"])
    marker = cache_dir.parent / ".blendit_coord" / f"prepare_{token[0]}.ready"
    prepared: list[dict[str, Any] | None] = [None]
    if distributed.is_main_process:
        result = prepare_data(config)
        prepared[0] = result.config
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ready\n", encoding="utf-8")
        print(
            "data preparation complete: "
            f"generated_splits={len(result.generated_splits)} "
            f"built_cache_files={result.built_cache_files} "
            f"removed_invalid_caches={len(result.removed_invalid_caches)}"
        )
    else:
        print(f"rank {distributed.rank}: waiting for rank 0 data preparation")
        while not marker.is_file():
            time.sleep(1.0)

    # This collective starts only after preparation has completed, so a long
    # cache build cannot consume the process group's collective timeout.
    dist.broadcast_object_list(prepared, src=0)
    if distributed.is_main_process:
        marker.unlink(missing_ok=True)
    if prepared[0] is None:
        raise RuntimeError("Failed to receive the prepared data configuration.")
    return prepared[0]


def cleanup_distributed(distributed: DistributedContext) -> None:
    if distributed.enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_cuda_out_of_memory(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "cudnn" in message)


def fail_fast_on_distributed_cuda_oom(
    exc: BaseException,
    distributed: DistributedContext,
    logger: Any | None = None,
    failure_log: str | Path | None = None,
) -> None:
    if not distributed.enabled or not is_cuda_out_of_memory(exc):
        return

    message = (
        "CUDA out of memory on "
        f"rank={distributed.rank} local_rank={distributed.local_rank}; "
        "exiting immediately so the launcher can terminate the remaining ranks"
    )
    try:
        if logger is not None:
            logger.critical(
                message,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        if failure_log is not None:
            failure_path = Path(failure_log)
            try:
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                with failure_path.open("a", encoding="utf-8") as stream:
                    print(message, file=stream)
                    traceback.print_exception(type(exc), exc, exc.__traceback__, file=stream)
                    stream.flush()
            except OSError as log_exc:
                print(
                    f"failed to write CUDA OOM log {failure_path}: {log_exc}",
                    file=sys.stderr,
                    flush=True,
                )
        print(message, file=sys.stderr, flush=True)
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        sys.stderr.flush()
    finally:
        # Avoid destroy_process_group(): after one rank OOMs, peers may already
        # be blocked in a different NCCL collective. A non-zero hard exit lets
        # torchrun observe the failure and terminate all other local ranks.
        os._exit(1)


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


def resolve_device(
    config: dict[str, Any],
    distributed: DistributedContext | None = None,
    *,
    enforce_cuda_requirement: bool = True,
) -> torch.device:
    distributed = distributed or DistributedContext()
    train_cfg = config["train"]
    requested = str(train_cfg.get("device", "cpu"))
    require_cuda = bool(train_cfg.get("require_cuda", False)) and enforce_cuda_requirement
    if require_cuda and not requested.startswith("cuda"):
        raise ValueError(
            "train.require_cuda=true requires train.device to be 'cuda' or 'cuda:<index>'."
        )
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError(
                "CUDA is required by train.require_cuda=true, but torch.cuda.is_available() is false."
            )
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


def _torch_load_checkpoint(path: str | Path, device: torch.device | str) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location=device)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> int:
    checkpoint = _torch_load_checkpoint(path, device)
    state = _normalize_state_dict_keys(checkpoint.get("model", checkpoint))
    unwrap_model(model).load_state_dict(state)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0))


def _checkpoint_model_state(path: str | Path, device: torch.device | str) -> dict[str, torch.Tensor]:
    checkpoint = _torch_load_checkpoint(path, device)
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


def configure_encoder_finetuning(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> EncoderFreezeResult:
    """Apply the configured encoder freeze strategy before optimizer creation."""
    target_model = unwrap_model(model)
    encoder = getattr(target_model, "encoder", None)
    if encoder is None:
        raise ValueError(f"Model {type(target_model).__name__} does not expose an encoder.")

    mode = str(config.get("train", {}).get("encoder_freeze_mode", "none")).lower()
    if mode not in {"none", "all", "partial"}:
        raise ValueError(
            "train.encoder_freeze_mode must be one of ['all', 'none', 'partial'], "
            f"got {mode!r}."
        )

    for parameter in encoder.parameters():
        parameter.requires_grad_(True)

    frozen_layers = 0
    if mode == "all":
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        frozen_layers = len(encoder.layers)
    elif mode == "partial":
        frozen_layers = int(config.get("train", {}).get("encoder_frozen_layers", 0))
        if not 0 < frozen_layers < len(encoder.layers):
            raise ValueError(
                "Partial encoder freezing requires 0 < train.encoder_frozen_layers "
                f"< model.num_layers ({len(encoder.layers)}), got {frozen_layers}."
            )
        stem_names = (
            "face_cont_proj",
            "surface_emb",
            "edge_cont_proj",
            "edge_type_emb",
            "edge_relation_emb",
            "input_norm",
            "edge_norm",
        )
        for name in stem_names:
            for parameter in getattr(encoder, name).parameters():
                parameter.requires_grad_(False)
        for layer in encoder.layers[:frozen_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad_(False)

    trainable_parameters = sum(parameter.numel() for parameter in target_model.parameters() if parameter.requires_grad)
    frozen_parameters = sum(parameter.numel() for parameter in target_model.parameters() if not parameter.requires_grad)
    return EncoderFreezeResult(
        mode=mode,
        frozen_layers=frozen_layers,
        trainable_parameters=trainable_parameters,
        frozen_parameters=frozen_parameters,
    )


def set_frozen_encoder_eval(model: torch.nn.Module, config: dict[str, Any]) -> None:
    """Keep dropout in frozen encoder components disabled during fine-tuning."""
    target_model = unwrap_model(model)
    encoder = getattr(target_model, "encoder", None)
    if encoder is None:
        return
    mode = str(config.get("train", {}).get("encoder_freeze_mode", "none")).lower()
    if mode == "all":
        encoder.eval()
    elif mode == "partial":
        frozen_layers = int(config.get("train", {}).get("encoder_frozen_layers", 0))
        stem_names = (
            "face_cont_proj",
            "surface_emb",
            "edge_cont_proj",
            "edge_type_emb",
            "edge_relation_emb",
            "input_norm",
            "edge_norm",
        )
        for name in stem_names:
            getattr(encoder, name).eval()
        for layer in encoder.layers[:frozen_layers]:
            layer.eval()


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.5f}" for key, value in sorted(metrics.items()))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in unwrap_model(model).parameters())


def log_config_summary(logger: Any, config: dict[str, Any]) -> None:
    data_cfg = config["data"]
    train_cfg = config["train"]
    model_cfg = config["model"]
    label_diffusion_cfg = config.get("label_diffusion", {})
    logger.info(
        "config data.cache_dir=%s data.cache_dirs=%s "
        "data.train_split=%s data.val_split=%s data.test_split=%s",
        data_cfg.get("cache_dir"),
        data_cfg.get("cache_dirs"),
        data_cfg.get("train_split"),
        data_cfg.get("val_split"),
        data_cfg.get("test_split"),
    )
    logger.info(
        "config train.epochs=%s train.batch_size=%s train.lr=%s train.weight_decay=%s "
        "train.resume=%s train.num_workers=%s train.require_cuda=%s "
        "train.distributed_timeout_seconds=%s",
        train_cfg.get("epochs"),
        train_cfg.get("batch_size"),
        train_cfg.get("lr"),
        train_cfg.get("weight_decay"),
        train_cfg.get("resume"),
        train_cfg.get("num_workers", 0),
        train_cfg.get("require_cuda", False),
        train_cfg.get("distributed_timeout_seconds", DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS),
    )
    logger.info(
        "config model.finetune_head=%s model.use_coarse_label_head=%s "
        "label_diffusion.prediction_type=%s train.encoder_freeze_mode=%s "
        "train.encoder_frozen_layers=%s",
        model_cfg.get("finetune_head", "n/a"),
        model_cfg.get("use_coarse_label_head", True),
        label_diffusion_cfg.get("prediction_type", "n/a"),
        train_cfg.get("encoder_freeze_mode", "none"),
        train_cfg.get("encoder_frozen_layers", 0),
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


def _nonfinite_message(name: str, sample_ids: list[str] | None = None) -> str:
    if sample_ids:
        return f"Non-finite values detected in {name}; sample_ids={sample_ids}"
    return f"Non-finite values detected in {name}"


def check_finite_tensor(name: str, tensor: torch.Tensor, sample_ids: list[str] | None = None) -> None:
    if tensor.numel() > 0 and not torch.isfinite(tensor).all():
        raise FloatingPointError(_nonfinite_message(name, sample_ids))


def check_finite_batch(batch: Any) -> None:
    sample_ids = getattr(batch, "sample_ids", None)
    check_finite_tensor("batch.face_cont", batch.face_cont, sample_ids)
    check_finite_tensor("batch.edge_cont", batch.edge_cont, sample_ids)


def check_finite_loss(loss: torch.Tensor, metrics: dict[str, float], sample_ids: list[str]) -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss detected; metrics={metrics}; sample_ids={sample_ids}")
    bad_metrics = {key: value for key, value in metrics.items() if not torch.isfinite(torch.tensor(value))}
    if bad_metrics:
        raise FloatingPointError(f"Non-finite metrics detected: {bad_metrics}; sample_ids={sample_ids}")
