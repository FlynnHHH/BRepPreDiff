from __future__ import annotations

import sys
from types import SimpleNamespace

from brepprediff.training.tracking import WandbTracker


class FakeRun:
    url = "https://wandb.example/run/test"

    def __init__(self) -> None:
        self.defined_metrics: list[tuple[str, dict]] = []
        self.logged: list[dict] = []
        self.finished = False

    def define_metric(self, name: str, **kwargs) -> None:
        self.defined_metrics.append((name, kwargs))

    def log(self, payload: dict) -> None:
        self.logged.append(payload)

    def finish(self) -> None:
        self.finished = True


class FakeLogger:
    def __init__(self) -> None:
        self.messages: list[tuple] = []

    def info(self, *args) -> None:
        self.messages.append(args)


def test_wandb_tracker_logs_train_and_validation_loss_curves(tmp_path, monkeypatch):
    fake_run = FakeRun()
    init_calls = []

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        return fake_run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    config = {
        "wandb": {
            "enabled": True,
            "project": "brepprediff-tests",
            "entity": "test-team",
            "tags": ["smoke"],
        },
        "train": {"epochs": 2},
    }

    tracker = WandbTracker.initialize(
        config,
        stage="finetune",
        run_dir=tmp_path / "run-name",
        logger=FakeLogger(),
        is_main_process=True,
    )
    tracker.log_epoch(
        2,
        {"total": 1.25, "cross_entropy": 1.0},
        {"total": 0.75, "acc": 0.8},
    )

    assert len(init_calls) == 1
    assert init_calls[0]["project"] == "brepprediff-tests"
    assert init_calls[0]["name"] == "run-name"
    assert init_calls[0]["job_type"] == "finetune"
    assert init_calls[0]["dir"] == str(tmp_path / "run-name")
    assert init_calls[0]["entity"] == "test-team"
    assert init_calls[0]["tags"] == ["smoke"]
    assert fake_run.defined_metrics == [
        ("epoch", {}),
        ("train/*", {"step_metric": "epoch"}),
        ("val/*", {"step_metric": "epoch"}),
    ]
    assert fake_run.logged == [
        {
            "epoch": 2,
            "train/total": 1.25,
            "train/cross_entropy": 1.0,
            "train/loss": 1.25,
            "val/total": 0.75,
            "val/acc": 0.8,
            "val/loss": 0.75,
        }
    ]

    tracker.finish()
    assert fake_run.finished
    assert tracker.run is None


def test_wandb_tracker_only_initializes_on_main_process(tmp_path, monkeypatch):
    def unexpected_init(**kwargs):
        raise AssertionError(f"wandb.init should not be called: {kwargs}")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=unexpected_init))

    tracker = WandbTracker.initialize(
        {},
        stage="pretrain",
        run_dir=tmp_path,
        logger=FakeLogger(),
        is_main_process=False,
    )

    assert tracker.run is None


def test_wandb_tracker_can_be_disabled(tmp_path, monkeypatch):
    def unexpected_init(**kwargs):
        raise AssertionError(f"wandb.init should not be called: {kwargs}")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=unexpected_init))

    tracker = WandbTracker.initialize(
        {"wandb": {"enabled": False}},
        stage="pretrain",
        run_dir=tmp_path,
        logger=FakeLogger(),
        is_main_process=True,
    )

    assert tracker.run is None
