from __future__ import annotations

from datetime import timedelta

import pytest

from blendit.training.common import (
    DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS,
    DistributedContext,
    distributed_timeout,
    fail_fast_on_distributed_cuda_oom,
    is_cuda_out_of_memory,
    resolve_device,
    setup_distributed,
)


def test_distributed_timeout_defaults_to_two_minutes():
    assert distributed_timeout({"train": {}}) == timedelta(
        seconds=DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS
    )


def test_distributed_timeout_rejects_non_positive_values():
    with pytest.raises(ValueError, match="must be positive"):
        distributed_timeout({"train": {"distributed_timeout_seconds": 0}})


def test_setup_distributed_passes_configured_timeout(monkeypatch):
    import torch.distributed as dist

    init_args = {}
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def record_init_process_group(**kwargs):
        init_args.update(kwargs)

    monkeypatch.setattr(dist, "init_process_group", record_init_process_group)

    distributed = setup_distributed(
        {"train": {"device": "cpu", "distributed_timeout_seconds": 75}}
    )

    assert distributed.enabled
    assert distributed.backend == "gloo"
    assert init_args == {"backend": "gloo", "timeout": timedelta(seconds=75)}


def test_resolve_device_can_require_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        resolve_device({"train": {"device": "cuda", "require_cuda": True}})


def test_resolve_device_rejects_require_cuda_with_cpu():
    with pytest.raises(ValueError, match="requires train.device"):
        resolve_device({"train": {"device": "cpu", "require_cuda": True}})


def test_setup_nccl_enables_async_error_handling_by_default(monkeypatch):
    import os
    import torch.distributed as dist

    init_args = {}
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("NCCL_ASYNC_ERROR_HANDLING", raising=False)
    monkeypatch.delenv("NCCL_BLOCKING_WAIT", raising=False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    monkeypatch.setattr("torch.cuda.set_device", lambda rank: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def record_init_process_group(**kwargs):
        init_args.update(kwargs)

    monkeypatch.setattr(dist, "init_process_group", record_init_process_group)

    distributed = setup_distributed(
        {"train": {"device": "cuda", "distributed_timeout_seconds": 90}}
    )

    assert distributed.backend == "nccl"
    assert os.environ["NCCL_ASYNC_ERROR_HANDLING"] == "1"
    assert init_args == {"backend": "nccl", "timeout": timedelta(seconds=90)}


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory. Tried to allocate 1.00 GiB",
        "RuntimeError: cuDNN error: out of memory",
    ],
)
def test_cuda_out_of_memory_detection(message):
    assert is_cuda_out_of_memory(RuntimeError(message))


def test_cuda_out_of_memory_detection_rejects_unrelated_error():
    assert not is_cuda_out_of_memory(RuntimeError("NCCL connection closed"))


def test_fail_fast_is_disabled_for_single_process(monkeypatch):
    def unexpected_exit(code):
        raise AssertionError(f"unexpected exit {code}")

    monkeypatch.setattr("os._exit", unexpected_exit)
    fail_fast_on_distributed_cuda_oom(
        RuntimeError("CUDA out of memory"),
        DistributedContext(),
    )


def test_fail_fast_hard_exits_distributed_worker(monkeypatch, tmp_path):
    class ExpectedExit(Exception):
        pass

    exit_codes = []

    def record_exit(code):
        exit_codes.append(code)
        raise ExpectedExit

    monkeypatch.setattr("os._exit", record_exit)

    failure_log = tmp_path / "logs" / "pretrain.oom.rank2.log"
    with pytest.raises(ExpectedExit):
        fail_fast_on_distributed_cuda_oom(
            RuntimeError("CUDA out of memory"),
            DistributedContext(enabled=True, rank=2, local_rank=2, world_size=4, backend="nccl"),
            failure_log=failure_log,
        )

    assert exit_codes == [1]
    contents = failure_log.read_text(encoding="utf-8")
    assert "rank=2 local_rank=2" in contents
    assert "CUDA out of memory" in contents
