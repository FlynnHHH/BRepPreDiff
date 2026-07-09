from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from blendit.config import feature_dims
from blendit.data import build_dataloader
from blendit.models import SegmentationModel, compute_segmentation_loss
from blendit.training.common import (
    MetricAverager,
    class_weights_from_config,
    cleanup_distributed,
    count_parameters,
    format_metrics,
    log_checkpoint_saved,
    log_config_summary,
    log_dataloader_summary,
    load_train_config,
    load_checkpoint,
    load_pretrain_checkpoint_for_finetune,
    load_pretrained_encoder,
    maybe_wrap_ddp,
    parse_train_args,
    prepare_run,
    resolve_device,
    save_checkpoint,
    setup_distributed,
)


def run_epoch(
    model,
    dataloader,
    optimizer,
    config,
    device,
    class_weights,
    train: bool,
    distributed,
) -> dict[str, float]:
    model.train(train)
    meter = MetricAverager()
    show_progress = distributed is None or distributed.is_main_process
    iterator = tqdm(
        dataloader,
        desc="train" if train else "val",
        leave=False,
        disable=not show_progress,
    )
    for batch in iterator:
        batch = batch.to(device)
        with torch.set_grad_enabled(train):
            logits = model(batch)
            loss, metrics = compute_segmentation_loss(logits, batch, config, class_weights)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_clip = float(config["train"].get("grad_clip_norm", 0.0) or 0.0)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        meter.update(metrics)
        if show_progress:
            iterator.set_postfix(total=f"{metrics['total']:.4f}", acc=f"{metrics['acc']:.3f}")
    return meter.compute(distributed=distributed, device=device)


def main() -> None:
    args = parse_train_args("Fine-tune B-Rep encoder for face segmentation.")
    config = load_train_config(args, stage="finetune")
    distributed = setup_distributed(config)
    try:
        config, run_dir, logger = prepare_run(args, stage="finetune", config=config, distributed=distributed)
        device = resolve_device(config, distributed)
        logger.info(
            "device=%s distributed=%s rank=%d local_rank=%d backend=%s",
            device,
            distributed.enabled,
            distributed.rank,
            distributed.local_rank,
            distributed.backend or "none",
        )
        log_config_summary(logger, config)

        face_dim, edge_dim = feature_dims(config)
        logger.info("building train dataloader")
        train_loader = build_dataloader(config, split="train", shuffle=True, distributed=distributed.enabled)
        log_dataloader_summary(logger, "train", train_loader, distributed)
        val_loader = None
        if config["data"].get("val_split"):
            logger.info("building val dataloader")
            val_loader = build_dataloader(config, split="val", shuffle=False, distributed=distributed.enabled)
            log_dataloader_summary(logger, "val", val_loader, distributed)
        else:
            logger.info("validation disabled: data.val_split is not set")

        logger.info("building segmentation model")
        model = SegmentationModel(config, face_dim, edge_dim).to(device)
        logger.info("model parameters=%d", count_parameters(model))
        resume = config["train"].get("resume")
        pretrain_checkpoint = config["train"].get("pretrain_checkpoint")
        pretrained = config["train"].get("pretrained_encoder")
        if pretrain_checkpoint and pretrained:
            raise ValueError("Set only one of train.pretrain_checkpoint and train.pretrained_encoder.")
        if resume and (pretrain_checkpoint or pretrained):
            logger.info("train.resume is set; skipping pretrain initialization")
        elif pretrain_checkpoint:
            logger.info("loading pretrain checkpoint for finetune=%s", pretrain_checkpoint)
            load_info = load_pretrain_checkpoint_for_finetune(pretrain_checkpoint, model=model, device=device)
            logger.info(
                "loaded pretrain checkpoint=%s tensors=%d skipped=%d",
                pretrain_checkpoint,
                len(load_info.loaded_keys),
                len(load_info.skipped_keys),
            )
        elif pretrained:
            logger.info("loading pretrained encoder=%s", pretrained)
            load_info = load_pretrained_encoder(pretrained, model=model, device=device)
            logger.info(
                "loaded pretrained encoder=%s tensors=%d skipped=%d",
                pretrained,
                len(load_info.loaded_keys),
                len(load_info.skipped_keys),
            )
        else:
            logger.info("no pretrained initialization configured")

        model = maybe_wrap_ddp(model, distributed)
        logger.info("model ddp_wrapped=%s", distributed.enabled)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["train"]["lr"]),
            weight_decay=float(config["train"]["weight_decay"]),
        )
        logger.info("optimizer ready: AdamW lr=%s weight_decay=%s", config["train"]["lr"], config["train"]["weight_decay"])
        class_weights = class_weights_from_config(config, device)
        logger.info("class_weights=%s", "configured" if class_weights is not None else "none")

        start_epoch = 0
        if resume:
            logger.info("loading resume checkpoint=%s", resume)
            start_epoch = load_checkpoint(resume, model=model, optimizer=optimizer, device=device)
            logger.info("resumed checkpoint=%s epoch=%d", resume, start_epoch)

        best_val = float("inf")
        epochs = int(config["train"]["epochs"])
        logger.info("training plan: start_epoch=%d target_epoch=%d total_epochs_to_run=%d", start_epoch, epochs, max(0, epochs - start_epoch))
        if start_epoch >= epochs:
            logger.info("checkpoint epoch=%d >= train.epochs=%d; no training steps to run", start_epoch, epochs)
            if distributed.is_main_process:
                path = run_dir / "checkpoints" / "last.pt"
                save_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    epoch=start_epoch,
                    config=config,
                    metrics={},
                )
                log_checkpoint_saved(logger, path, start_epoch)
            return

        for epoch in range(start_epoch + 1, epochs + 1):
            logger.info("epoch=%d/%d split=train start", epoch, epochs)
            train_sampler = getattr(train_loader, "sampler", None)
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                config,
                device,
                class_weights,
                train=True,
                distributed=distributed,
            )
            logger.info("epoch=%d split=train %s", epoch, format_metrics(train_metrics))

            val_metrics = None
            if val_loader is not None and epoch % int(config["train"]["validate_every_epochs"]) == 0:
                logger.info("epoch=%d/%d split=val start", epoch, epochs)
                val_sampler = getattr(val_loader, "sampler", None)
                if hasattr(val_sampler, "set_epoch"):
                    val_sampler.set_epoch(epoch)

                val_metrics = run_epoch(
                    model,
                    val_loader,
                    optimizer,
                    config,
                    device,
                    class_weights,
                    train=False,
                    distributed=distributed,
                )
                logger.info("epoch=%d split=val %s", epoch, format_metrics(val_metrics))
                if val_metrics.get("total", float("inf")) < best_val:
                    best_val = val_metrics["total"]
                    if distributed.is_main_process:
                        path = run_dir / "checkpoints" / "best.pt"
                        save_checkpoint(
                            path,
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            config=config,
                            metrics=val_metrics,
                        )
                        log_checkpoint_saved(logger, path, epoch)

            if distributed.is_main_process and epoch % int(config["run"]["save_every_epochs"]) == 0:
                metrics = val_metrics or train_metrics
                path = Path(run_dir) / "checkpoints" / f"epoch_{epoch:04d}.pt"
                save_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    metrics=metrics,
                )
                log_checkpoint_saved(logger, path, epoch)

        if distributed.is_main_process:
            path = run_dir / "checkpoints" / "last.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                epoch=epochs,
                config=config,
                metrics=train_metrics,
            )
            log_checkpoint_saved(logger, path, epochs)
        logger.info("finished fine-tuning")
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
