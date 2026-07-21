from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")


def _load_ablation_suite_module():
    script_path = Path(__file__).parents[1] / "scripts" / "run_ablation_suite.py"
    spec = importlib.util.spec_from_file_location("blendit_ablation_suite", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ablation_suite_has_one_default_baseline_and_labels_every_alternative():
    module = _load_ablation_suite_module()
    experiments = module.experiments("runs/pretrain/example")
    finetunes = [experiment for experiment in experiments if experiment.stage == "finetune"]
    baselines = [experiment for experiment in finetunes if experiment.factors["role"] == "baseline"]

    assert len(baselines) == 1
    assert baselines[0].factors == {
        "role": "baseline",
        "ablated_factor": "none",
        "coarse_label": True,
        "head": "diffloss",
        "prediction_type": "x_start_epsilon",
        "x_start_loss_weight": 1.0,
        "epsilon_loss_weight": 0.5,
        "encoder_freeze": "none",
    }

    ablations = [experiment for experiment in experiments if experiment not in baselines]
    for experiment in ablations:
        if experiment.name == "baseline_pretrain":
            continue
        assert experiment.factors["role"] == "ablation"
        assert experiment.factors["ablated_factor"] != "none"
        assert "ablation" in experiment.name
        assert "ablation" in experiment.run_name
