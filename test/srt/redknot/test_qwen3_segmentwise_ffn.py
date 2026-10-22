from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from test_segpaged_workflow import _isolated_imports


class TestQwen3SegmentwiseFFN(unittest.TestCase):
    def setUp(self):
        imports = _isolated_imports()
        self.runtime = imports.__enter__()
        self.addCleanup(imports.__exit__, None, None, None)

    def test_dense_prefix_and_suffix_are_fail_closed(self):
        schedule = self.runtime.driver.SparseFFNSchedule(
            dense_until=3,
            dense_suffix_layers=2,
            total_layers=64,
            mass_thresh=0.5,
            deep_layer_start=40,
            mass_thresh_deep=0.35,
        )
        self.assertEqual(
            [i for i in range(64) if schedule.is_dense_layer(i)],
            [0, 1, 2, 62, 63],
        )

    def test_each_document_gets_an_independent_budget_but_one_ffn_call(self):
        schedule = self.runtime.driver.SparseFFNSchedule(
            dense_until=0,
            mass_thresh=0.5,
            deep_layer_start=64,
            recent_n=1,
            min_keep=1,
        )
        hidden = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
        # A global top-p would spend almost all mass on document zero.  The
        # segmentwise contract must still select rows from document one.
        importance = torch.tensor([[100.0, 90.0, 80.0, 1.0, 1.0, 1.0]])
        calls = []

        def ffn(rows):
            calls.append(rows.clone())
            return torch.ones_like(rows)

        out, stats = self.runtime.driver.apply_sparse_ffn(
            hidden,
            ffn,
            layer_idx=3,
            schedule=schedule,
            importance=importance,
            segment_lengths=(3, 3),
            return_stats=True,
        )
        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(stats["selected"], 3)
        delta = out - hidden
        self.assertTrue(bool((delta[0, 3:] != 0).any()))
        self.assertTrue(bool((delta[0, 2] != 0).all()))
        self.assertTrue(bool((delta[0, 5] != 0).all()))

    def test_bad_segment_geometry_fails_before_ffn(self):
        schedule = self.runtime.driver.SparseFFNSchedule(
            dense_until=0, mass_thresh=0.5, recent_n=0,
        )
        called = False

        def ffn(rows):
            nonlocal called
            called = True
            return rows

        with self.assertRaisesRegex(ValueError, "tile L exactly"):
            self.runtime.driver.apply_sparse_ffn(
                torch.zeros(1, 6, 4),
                ffn,
                layer_idx=3,
                schedule=schedule,
                importance=torch.ones(1, 6),
                segment_lengths=(3, 2),
            )
        self.assertFalse(called)

    def test_unselected_rows_reuse_offline_delta(self):
        schedule = self.runtime.driver.SparseFFNSchedule(
            dense_until=0, mass_thresh=0.1, deep_layer_start=64, recent_n=0
        )
        hidden = torch.zeros(1, 4, 3)
        fallback = torch.full_like(hidden, 7.0)
        importance = torch.tensor([[100.0, 1.0, 100.0, 1.0]])

        def ffn(rows):
            return torch.full_like(rows, 11.0)

        out, stats = self.runtime.driver.apply_sparse_ffn(
            hidden,
            ffn,
            layer_idx=3,
            schedule=schedule,
            importance=importance,
            segment_lengths=(2, 2),
            fallback_delta=fallback,
            return_stats=True,
        )
        self.assertEqual(stats["selected"], 2)
        self.assertEqual(stats["reused"], 2)
        self.assertTrue(torch.equal(out[0, 0], torch.full((3,), 11.0)))
        self.assertTrue(torch.equal(out[0, 1], torch.full((3,), 7.0)))
        self.assertTrue(torch.equal(out[0, 2], torch.full((3,), 11.0)))
        self.assertTrue(torch.equal(out[0, 3], torch.full((3,), 7.0)))

    def test_head_profile_enforces_3_59_2(self):
        profile = {
            "num_layers": 64,
            "num_kv_heads": 8,
            "dense_prefix_layers": 3,
            "dense_suffix_layers": 2,
            "kv_head_classification": [["local"] * 8 for _ in range(64)],
            "kv_head_max_distance": [[4096] * 8 for _ in range(64)],
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "profile.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            config = self.runtime.config.HeadClassConfig.from_json(str(path))
        dense_layers = [
            layer
            for layer in range(64)
            if all(config.get_strategy(layer, h).is_dense() for h in range(8))
        ]
        self.assertEqual(dense_layers, [0, 1, 2, 62, 63])


if __name__ == "__main__":
    unittest.main()
