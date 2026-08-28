from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from blendit.config import feature_dims
from blendit.data import build_dataloader
from blendit.data.load_data import split_has_items
from blendit.models import DiffusionPretrainModel, DiffusionSchedule, compute_pretrain_loss
from blendit.training.common import (
    MetricAverager,
    check_finite_batch,
    check_finite_loss,
    cleanup_distributed,
    count_parameters,
    fail_fast_on_distributed_cuda_oom,
    format_metrics,
    log_checkpoint_saved,
    log_config_summary,
    log_dataloader_summary,
    load_train_config,
    load_checkpoint,
    maybe_wrap_ddp,
    parse_train_args,
    prepare_training_data,
    prepare_run,
    resolve_device,
    save_checkpoint,
    setup_distributed,
)
from blendit.training.tracking import WandbTracker


def run_epoch(
    model,
    schedule,
    dataloader,
    optimizer,
    config,
    device,
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
        if train:
            optimizer.zero_grad(set_to_none=True)
        batch = batch.to(device)
        check_finite_batch(batch)
        num_graphs = int(batch.graph_ptr.numel() - 1)
        graph_t = torch.randint(0, schedule.timesteps, (num_graphs,), dtype=torch.long, device=device)
        face_t = graph_t[batch.batch_index]
        edge_t = graph_t[batch.edge_batch_index] if batch.edge_batch_index.numel() else batch.edge_batch_index

        face_noisy, face_noise = schedule.q_sample(batch.face_cont, face_t)
        if batch.edge_cont.numel() > 0:
            edge_noisy, edge_noise = schedule.q_sample(batch.edge_cont, edge_t)
        else:
            edge_noisy = batch.edge_cont
            edge_noise = torch.empty_like(batch.edge_cont)

        with torch.set_grad_enabled(train):
            outputs = model(batch, face_noisy, edge_noisy, face_t)
            loss, metrics = compute_pretrain_loss(
                outputs,
                batch,
                face_noise=face_noise,
                edge_noise=edge_noise,
                config=config,
            )
            check_finite_loss(loss, metrics, batch.sample_ids)
            if train:
                loss.backward()
                grad_clip = float(config["train"].get("grad_clip_norm", 0.0) or 0.0)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
        meter.update(metrics)
        if show_progress:
            iterator.set_postfix(total=f"{metrics['total']:.4f}")
    return meter.compute(distributed=distributed, device=device)


def main() -> None:
    args = parse_train_args("Diffusion pretraining for B-Rep graph encoder.")
    config = load_train_config(args, stage="pretrain")
    distributed = setup_distributed(config)
    logger = None
    run_dir = None
    tracker = WandbTracker()
    try:
        config = prepare_training_data(config, distributed)
        config, run_dir, logger = prepare_run(args, stage="pretrain", config=config, distributed=distributed)
        tracker = WandbTracker.initialize(
            config,
            stage="pretrain",
            run_dir=run_dir,
            logger=logger,
            is_main_process=distributed.is_main_process,
        )
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
        if split_has_items(config, "val"):
            logger.info("building val dataloader")
            val_loader = build_dataloader(config, split="val", shuffle=False, distributed=distributed.enabled)
            log_dataloader_summary(logger, "val", val_loader, distributed)
        else:
            logger.info("validation disabled: data.val_split is not set or is empty")

        logger.info("building pretrain model")
        model = DiffusionPretrainModel(config, face_dim, edge_dim).to(device)
        schedule = DiffusionSchedule(
            int(config["diffusion"]["timesteps"]),
            float(config["diffusion"]["beta_start"]),
            float(config["diffusion"]["beta_end"]),
        ).to(device)
        logger.info(
            "diffusion schedule ready: timesteps=%d beta_start=%s beta_end=%s",
            schedule.timesteps,
            config["diffusion"]["beta_start"],
            config["diffusion"]["beta_end"],
        )
        logger.info("model parameters=%d", count_parameters(model))
        model = maybe_wrap_ddp(model, distributed)
        logger.info("model ddp_wrapped=%s", distributed.enabled)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["train"]["lr"]),
            weight_decay=float(config["train"]["weight_decay"]),
        )
        logger.info("optimizer ready: AdamW lr=%s weight_decay=%s", config["train"]["lr"], config["train"]["weight_decay"])

        start_epoch = 0
        resume = config["train"].get("resume")
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
                schedule,
                train_loader,
                optimizer,
                config,
                device,
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
                    schedule,
                    val_loader,
                    optimizer,
                    config,
                    device,
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

            tracker.log_epoch(epoch, train_metrics, val_metrics)

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
        logger.info("finished pretraining")
    except RuntimeError as exc:
        failure_log = (
            run_dir / "logs" / f"pretrain.oom.rank{distributed.rank}.log"
            if run_dir is not None
            else None
        )
        fail_fast_on_distributed_cuda_oom(
            exc,
            distributed,
            logger,
            failure_log=failure_log,
        )
        raise
    finally:
        try:
            tracker.finish()
        finally:
            cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
