#!/usr/bin/env python3
"""Smoke tests for variable-node batching invariants."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml2_meta_causal_discovery.utils.datautils import (  # noqa: E402
    SameNodeCountBatchSampler,
    transformer_classifier_split_variable_nodes,
)


class DummyVariableNodeDataset:
    def __init__(self, node_counts):
        self.node_counts = list(node_counts)

    def __len__(self):
        return len(self.node_counts)

    def node_count(self, index):
        return self.node_counts[index]


class VariableNodeBatchingSmokeTest(unittest.TestCase):
    def test_sampler_never_mixes_node_counts(self) -> None:
        dataset = DummyVariableNodeDataset([3, 5, 3, 4, 5, 3, 4, 5])
        sampler = SameNodeCountBatchSampler(
            dataset,
            batch_size=2,
            shuffle=False,
        )
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        for batch in batches:
            node_counts = {dataset.node_count(index) for index in batch}
            self.assertEqual(len(node_counts), 1)

    def test_collator_shares_sample_count_and_row_indices(self) -> None:
        num_observations = 50
        num_nodes = 5
        row_ids = np.arange(num_observations, dtype=np.float32)[:, None]
        base = np.repeat(row_ids, num_nodes, axis=1)
        graph = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        batch = [
            (base.copy(), graph.copy()),
            (base.copy() + 100.0, graph.copy()),
            (base.copy() + 200.0, graph.copy()),
        ]
        collate = transformer_classifier_split_variable_nodes(17, 17)
        inputs, targets, mask = collate(batch)
        self.assertEqual(tuple(inputs.shape), (3, 17, 5))
        self.assertEqual(tuple(targets.shape), (3, 5, 5))
        self.assertIsNone(mask)
        np.testing.assert_allclose(
            inputs[1, :, 0].numpy() - inputs[0, :, 0].numpy(),
            100.0,
        )
        np.testing.assert_allclose(
            inputs[2, :, 0].numpy() - inputs[0, :, 0].numpy(),
            200.0,
        )

    def test_collator_rejects_mixed_node_counts(self) -> None:
        collate = transformer_classifier_split_variable_nodes(10, 10)
        graph3 = np.zeros((3, 3), dtype=np.float32)
        graph4 = np.zeros((4, 4), dtype=np.float32)
        batch = [
            (np.zeros((20, 3), dtype=np.float32), graph3),
            (np.zeros((20, 4), dtype=np.float32), graph4),
        ]
        with self.assertRaisesRegex(ValueError, "same node count"):
            collate(batch)


if __name__ == "__main__":
    unittest.main()
