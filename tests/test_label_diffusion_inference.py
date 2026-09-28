"""Compatibility checks against predictions recorded with the archived inference code."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest

import torch

from brepprediff.config import feature_dims
from brepprediff.data.graph import BRepGraph, collate_graphs
from brepprediff.models import build_finetune_model, predict_finetune_probabilities
from brepprediff.training.common import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def example_graphs(face_dim: int, edge_dim: int) -> list[BRepGraph]:
    graphs = []
    for index, count in enumerate((2, 3)):
        src = torch.arange(count - 1)
        dst = src + 1
        edge_index = torch.stack((torch.cat((src, dst)), torch.cat((dst, src))))
        edges = edge_index.shape[1]
        graphs.append(BRepGraph(
            face_cont=torch.linspace(-1.0, 1.0, count * face_dim).reshape(count, face_dim),
            face_surface_type=torch.arange(count) % 3,
            edge_index=edge_index,
            edge_cont=torch.linspace(-0.5, 0.5, edges * edge_dim).reshape(edges, edge_dim),
            edge_type=torch.arange(edges) % 4,
            edge_relation=torch.arange(edges) % 2,
            labels=None,
            sample_id=f"compatibility_{index}",
        ))
    return graphs


class LabelDiffusionInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)
        cls.checkpoint = ROOT / "ckpt/cadsynth_best.pt"
        payload = torch.load(cls.checkpoint, map_location="cpu", weights_only=True)
        cls.config = payload["config"]
        cls.model = build_finetune_model(cls.config, *feature_dims(cls.config)).eval()
        load_checkpoint(cls.checkpoint, model=cls.model, device="cpu")
        cls.graphs = example_graphs(*feature_dims(cls.config))
        cls.reference = json.loads(
            (ROOT / "tests/fixtures/cadsynth_one_step_probabilities.json").read_text()
        )

    def test_matches_archived_predictions(self) -> None:
        self.assertEqual(
            hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            self.reference["checkpoint_sha256"],
        )
        batch = collate_graphs(self.graphs)
        self.assertEqual(batch.sample_ids, self.reference["sample_ids"])
        with torch.no_grad():
            actual = predict_finetune_probabilities(self.model, batch, self.config)
        expected = torch.tensor(self.reference["probabilities"], dtype=actual.dtype)
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)

    def test_predictions_do_not_depend_on_batch_partition(self) -> None:
        with torch.no_grad():
            combined = predict_finetune_probabilities(
                self.model, collate_graphs(self.graphs), self.config,
            )
            separate = torch.cat([
                predict_finetune_probabilities(self.model, collate_graphs([graph]), self.config)
                for graph in self.graphs
            ])
        torch.testing.assert_close(combined, separate, rtol=1.0e-5, atol=1.0e-6)

    def test_unsupported_sampling_is_rejected(self) -> None:
        for change in ({"sampling_steps": 2}, {"prediction_type": "epsilon"}, {"ddim_eta": 0.5}):
            with self.subTest(change=change):
                config = copy.deepcopy(self.config)
                config["label_diffusion"].update(change)
                with self.assertRaisesRegex(ValueError, "Archived diffusion inference requires"):
                    build_finetune_model(config, *feature_dims(config))

    def test_training_has_an_explicit_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "evaluation and inference only"):
            self.model(collate_graphs(self.graphs))


if __name__ == "__main__":
    unittest.main()
