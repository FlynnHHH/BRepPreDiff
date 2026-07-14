from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

from blendit.config import feature_dims
from blendit.data import build_dataloader
from blendit.models import (
    DiffusionSegmentationModel,
    build_segmentation_model,
    compute_label_diffusion_loss,
    compute_segmentation_loss,
    predict_segmentation_probabilities,
    prepare_label_diffusion_training_batch,
    segmentation_confusion_matrix,
    segmentation_metrics_from_confusion_matrix,
    segmentation_metrics_from_probabilities,
)
from blendit.training.common import (
    MetricAverager,
    class_weights_from_config,
    check_finite_batch,
    check_finite_loss,
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
    unwrap_model,
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
    num_classes = int(config["model"]["num_classes"])
    validation_confusion = (
        None
        if train
        else torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    )
    show_progress = distributed is None or distributed.is_main_process
    iterator = tqdm(
        dataloader,
        desc="train" if train else "val",
        leave=False,
        disable=not show_progress,
    )
    for batch in iterator:
        batch = batch.to(device)
        check_finite_batch(batch)
        with torch.set_grad_enabled(train):
            target_model = unwrap_model(model)
            if isinstance(target_model, DiffusionSegmentationModel):
                prepared = prepare_label_diffusion_training_batch(target_model, batch, config)
                noise_prediction = model(
                    batch,
                    prepared.x_t,
                    prepared.timesteps,
                    prepared.face_indices,
                )
                loss, metrics = compute_label_diffusion_loss(
                    noise_prediction,
                    prepared,
                    target_model,
                    class_weights,
                )
                if not train:
                    probabilities = predict_segmentation_probabilities(target_model, batch, config)
                    metrics.update(segmentation_metrics_from_probabilities(probabilities, batch, config))
            else:
                logits = model(batch)
                loss, metrics = compute_segmentation_loss(logits, batch, config, class_weights)
                if not train:
                    probabilities = logits.softmax(dim=-1)
                    metrics.update(segmentation_metrics_from_probabilities(probabilities, batch, config))
            if validation_confusion is not None:
                validation_confusion.add_(segmentation_confusion_matrix(probabilities, batch, config))
            check_finite_loss(loss, metrics, batch.sample_ids)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_clip = float(config["train"].get("grad_clip_norm", 0.0) or 0.0)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
        meter.update(metrics)
        if show_progress:
            iterator.set_postfix(total=f"{metrics['total']:.4f}", acc=f"{metrics['acc']:.3f}")
    epoch_metrics = meter.compute(distributed=distributed, device=device)
    if validation_confusion is not None:
        if distributed is not None and distributed.enabled:
            dist.all_reduce(validation_confusion, op=dist.ReduceOp.SUM)
        epoch_metrics.update(segmentation_metrics_from_confusion_matrix(validation_confusion))
    return epoch_metrics


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

        head_type = str(config.get("model", {}).get("finetune_head", "mlp"))
        logger.info("building segmentation model head=%s", head_type)
        model = build_segmentation_model(config, face_dim, edge_dim).to(device)
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

        best_f1 = float("-inf")
        logger.info("best checkpoint selection metric=validation macro-F1 mode=max")
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
                selection_value = val_metrics.get("f1", float("-inf"))
                if selection_value > best_f1:
                    best_f1 = selection_value
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
