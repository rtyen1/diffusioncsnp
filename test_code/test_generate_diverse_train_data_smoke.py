#!/usr/bin/env python3
"""Small deterministic smoke tests for diverse causal data generation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_diverse_train_data import (
    ACTIVATION_NAMES,
    METADATA_DTYPE,
    MLP_DEPTHS,
    MLP_HIDDEN_WIDTHS,
    RFF_FEATURE_COUNTS,
    SEM_GP,
    SEM_LINEAR,
    SEM_MLP,
    SEM_MLP_NONADDITIVE,
    SEM_RFF,
    allocate_node_count_datasets,
    distribution_name,
    generate_dataset,
    generate_parametric_sem,
    validate_probabilities,
    write_shard,
)
from ml2_meta_causal_discovery.utils.datautils import (
    MultipleFileDataset,
    SameNodeCountBatchSampler,
    transformer_classifier_split_variable_nodes,
)


class DiverseGenerationSmokeTest(unittest.TestCase):
    def test_gp_output_uses_common_standardisation(self) -> None:
        probabilities = np.zeros(5, dtype=np.float64)
        probabilities[SEM_GP] = 1.0
        data, graph, metadata = generate_dataset(
            num_nodes=3,
            num_samples=64,
            sem_probs=probabilities,
            dataset_seed=271828,
        )
        self.assertEqual(data.shape, (64, 3))
        self.assertEqual(graph.shape, (3, 3))
        self.assertEqual(int(metadata["sem_type"]), SEM_GP)
        self.assertTrue(np.isfinite(data).all())
        np.testing.assert_allclose(data.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(data.std(axis=0), 1.0, atol=1e-5)

    def test_d3_to_d20_hdf5_and_training_batch_contract(self) -> None:
        sem_types = (SEM_LINEAR, SEM_MLP, SEM_MLP_NONADDITIVE, SEM_RFF)
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = []
            for num_nodes in range(3, 21):
                sem_type = sem_types[(num_nodes - 3) % len(sem_types)]
                probabilities = np.zeros(5, dtype=np.float64)
                probabilities[sem_type] = 1.0
                path = Path(temp_dir) / f"d{num_nodes}.hdf5"
                write_shard(
                    output_path=path,
                    num_nodes=num_nodes,
                    num_samples=32,
                    num_datasets=2,
                    sem_probs=probabilities,
                    shard_seed=5000 + num_nodes,
                    compression="none",
                    overwrite=False,
                    show_progress=False,
                )
                paths.append(path)

            dataset = MultipleFileDataset(paths)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=SameNodeCountBatchSampler(
                    dataset,
                    batch_size=2,
                    shuffle=False,
                ),
                collate_fn=transformer_classifier_split_variable_nodes(16, 16),
                num_workers=0,
            )
            seen_node_counts = []
            for inputs, targets, mask in loader:
                num_nodes = inputs.shape[-1]
                self.assertEqual(tuple(inputs.shape), (2, 16, num_nodes))
                self.assertEqual(tuple(targets.shape), (2, num_nodes, num_nodes))
                self.assertIsNone(mask)
                seen_node_counts.append(num_nodes)
            self.assertEqual(seen_node_counts, list(range(3, 21)))

    def test_node_count_probability_allocation(self) -> None:
        counts = allocate_node_count_datasets(
            node_counts=[3, 4, 5],
            num_datasets_total=11,
            probabilities=[0.2, 0.3, 0.5],
        )
        self.assertEqual(counts, {3: 2, 4: 3, 5: 6})
        self.assertEqual(sum(counts.values()), 11)

    def test_new_sem_seed_reproducibility(self) -> None:
        for sem_type in (SEM_MLP, SEM_MLP_NONADDITIVE, SEM_RFF):
            with self.subTest(sem_type=sem_type):
                probabilities = np.zeros(5, dtype=np.float64)
                probabilities[sem_type] = 1.0
                first = generate_dataset(
                    num_nodes=6,
                    num_samples=96,
                    sem_probs=probabilities,
                    dataset_seed=314159 + sem_type,
                )
                second = generate_dataset(
                    num_nodes=6,
                    num_samples=96,
                    sem_probs=probabilities,
                    dataset_seed=314159 + sem_type,
                )
                np.testing.assert_array_equal(first[0], second[0])
                np.testing.assert_array_equal(first[1], second[1])
                self.assertEqual(first[2].tobytes(), second[2].tobytes())

    def test_new_parametric_sem_families(self) -> None:
        dag = np.zeros((4, 4), dtype=np.int8)
        dag[0, 1] = 1
        dag[1, 2] = 1
        dag[2, 3] = 1

        for sem_type in (SEM_MLP, SEM_MLP_NONADDITIVE, SEM_RFF):
            with self.subTest(sem_type=sem_type):
                data, details = generate_parametric_sem(
                    dag_topological=dag,
                    num_samples=128,
                    sem_type=sem_type,
                    rng=np.random.default_rng(1000 + sem_type),
                )
                self.assertEqual(data.shape, (128, 4))
                self.assertTrue(np.isfinite(data).all())
                np.testing.assert_allclose(data.mean(axis=0), 0.0, atol=1e-5)
                np.testing.assert_allclose(data.std(axis=0), 1.0, atol=1e-5)

                if sem_type in (SEM_MLP, SEM_MLP_NONADDITIVE):
                    self.assertIn(details.mlp_depth, MLP_DEPTHS)
                    self.assertIn(details.mlp_hidden_width, MLP_HIDDEN_WIDTHS)
                    self.assertIn(details.activation, ACTIVATION_NAMES)
                if sem_type == SEM_MLP_NONADDITIVE:
                    self.assertTrue(
                        np.isfinite(details.mean_nonadditive_noise_scale)
                    )
                    self.assertFalse(np.isfinite(details.mean_target_r2))
                else:
                    self.assertTrue(np.isfinite(details.mean_target_r2))
                if sem_type == SEM_RFF:
                    self.assertIn(details.rff_num_features, RFF_FEATURE_COUNTS)
                    self.assertTrue(np.isfinite(details.rff_lengthscale))

    def test_hdf5_contract_and_legacy_probability_defaults(self) -> None:
        legacy_probs = validate_probabilities(0.35, 0.50, 0.15)
        np.testing.assert_allclose(legacy_probs, [0.35, 0.50, 0.15, 0.0, 0.0])
        self.assertEqual(distribution_name(legacy_probs), "diverse_causal_pretrain_v2")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for sem_type in (SEM_MLP, SEM_MLP_NONADDITIVE, SEM_RFF):
                probabilities = np.zeros(5, dtype=np.float64)
                probabilities[sem_type] = 1.0
                output_path = root / f"sem_{sem_type}.hdf5"
                write_shard(
                    output_path=output_path,
                    num_nodes=4,
                    num_samples=64,
                    num_datasets=3,
                    sem_probs=probabilities,
                    shard_seed=2000 + sem_type,
                    compression="none",
                    overwrite=False,
                    show_progress=False,
                )
                with h5py.File(output_path, "r") as handle:
                    self.assertEqual(handle["data"].shape, (3, 64, 4))
                    self.assertEqual(handle["label"].shape, (3, 4, 4))
                    self.assertEqual(handle["metadata"].shape, (3,))
                    self.assertEqual(
                        handle["metadata"].dtype.names,
                        METADATA_DTYPE.names,
                    )
                    self.assertEqual(
                        handle.attrs["distribution"],
                        "diverse_causal_pretrain_v2",
                    )
                    self.assertTrue(
                        np.all(handle["metadata"]["sem_type"][:] == sem_type)
                    )
                    self.assertEqual(
                        np.unique(handle["metadata"]["dataset_seed"][:]).size,
                        3,
                    )


if __name__ == "__main__":
    unittest.main()
