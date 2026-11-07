"""Fast tests of the Station1 data contract and resumable generation (no R fit)."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import networkx as nx
import numpy as np
import pandas as pd

from test_code.realworldtest import generate_causalassembly_station1 as gen


COLUMNS = [f"Station1_mp_{i}" for i in range(6)]
DAG = np.zeros((6, 6), dtype=np.int8)
for i, j in [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]:
    DAG[i, j] = 1


class FakeCell:
    def __init__(self, fail_on_call=None):
        self.random_state = np.random.default_rng(42)
        self.calls = 0
        self.fail_on_call = fail_on_call

    def sample_from_drf(self, size):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated interrupted dataset")
        values = 100 + self.random_state.normal(size=(size, 6)) * np.arange(1, 7)[None]
        return pd.DataFrame(values, columns=COLUMNS)


def protocol(count=100, sizes=gen.SAMPLE_SIZES):
    return {"columns": COLUMNS, "datasets_per_cell": count, "sample_sizes": list(sizes),
            "sampling_seed": 42, "sampling_batch_rows": 256,
            "label_convention": "label[i, j] = 1 means i -> j"}


class Station1GeneratorTests(unittest.TestCase):
    def test_seeds_unique_stable_and_not_paired(self):
        seeds = [gen.dataset_seed(42, n, r) for n in gen.SAMPLE_SIZES for r in range(100)]
        self.assertEqual(len(set(seeds)), 600)
        self.assertEqual(seeds, [gen.dataset_seed(42, n, r) for n in gen.SAMPLE_SIZES for r in range(100)])
        self.assertNotEqual(gen.dataset_seed(42, 20, 0), gen.dataset_seed(42, 50, 0))

    def test_exact_600_dataset_layout(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            config = protocol()
            cell = FakeCell()
            for n in gen.SAMPLE_SIZES:
                gen.generate_cell(gen.shard_path(root, n), config, n, DAG, cell)
            report = gen.validate_benchmark(root, config, DAG)
            self.assertEqual(report["total_datasets"], 600)
            self.assertEqual(len(report["cells"]), 6)
            for n in gen.SAMPLE_SIZES:
                path = gen.shard_path(root, n)
                with h5py.File(path, "r") as handle:
                    self.assertEqual(handle["data"].shape, (100, n, 6))
                    self.assertEqual(handle["label"].shape, (100, 6, 6))
                    self.assertTrue(np.array_equal(handle["label"][0], DAG))
                    self.assertGreater(handle["data"][0].mean(), 90)  # No added standardization.
                    self.assertFalse(np.array_equal(handle["data"][0], handle["data"][1]))
            before = cell.calls
            gen.generate_cell(gen.shard_path(root, 20), config, 20, DAG, cell)
            self.assertEqual(before, cell.calls)

    def test_unified_manifest_compatibility_when_available(self):
        try:
            from test_code.unified_benchmark.build_manifest import build_manifest
        except ModuleNotFoundError:
            self.skipTest("The optional unified benchmark evaluator is not installed here")
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            config = protocol(count=2, sizes=(20, 50))
            for n in (20, 50):
                gen.generate_cell(gen.shard_path(root, n), config, n, DAG, FakeCell())
            rows = build_manifest(root, selection_seed=42, strict_layout=False)
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["sem_type"] for row in rows}, {"causalassembly_drf"})
            self.assertEqual({row["noise_mode"] for row in rows}, {"observational_drf"})

    def test_interruption_reuses_finished_rows_and_matches_clean_run(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            config = protocol(count=3, sizes=(20,))
            path = gen.shard_path(root / "resume", 20)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                gen.generate_cell(path, config, 20, DAG, FakeCell(fail_on_call=2))
            self.assertFalse(path.exists())
            self.assertEqual(list((root / "resume").glob("*/d*/n*/test/d*_shard_*.hdf5")), [])
            with h5py.File(gen.partial_path(path), "r") as handle:
                self.assertEqual(handle["completed"][:].tolist(), [True, False, False])
                first = handle["data"][0]
            resumed = FakeCell()
            gen.generate_cell(path, config, 20, DAG, resumed)
            self.assertEqual(resumed.calls, 2)
            clean = gen.shard_path(root / "clean", 20)
            gen.generate_cell(clean, config, 20, DAG, FakeCell())
            with h5py.File(path, "r") as a, h5py.File(clean, "r") as b:
                np.testing.assert_array_equal(a["data"][:], b["data"][:])
                np.testing.assert_array_equal(a["data"][0], first)

    def test_chunking_is_reproducible_and_never_exceeds_batch_limit(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            config = protocol(count=1, sizes=(1000,))
            cell = FakeCell()
            path = gen.shard_path(Path(temporary), 1000)
            gen.generate_cell(path, config, 1000, DAG, cell)
            self.assertEqual(cell.calls, 4)

    def test_wrong_protocol_and_corruption_are_rejected_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            config = protocol(count=2, sizes=(20,))
            path = gen.shard_path(Path(temporary), 20)
            gen.generate_cell(path, config, 20, DAG, FakeCell())
            changed = {**config, "sampling_seed": 43}
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                gen.generate_cell(path, changed, 20, DAG, FakeCell())
            with h5py.File(path, "r+") as handle:
                handle["data"][0, 0, 0] += 1
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                gen.validate_shard(path, config, 20, DAG)

    def test_nonfinite_or_wrong_order_does_not_commit_dataset(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            config = protocol(count=1, sizes=(20,))
            path = gen.shard_path(Path(temporary), 20)
            bad = FakeCell()
            bad.sample_from_drf = lambda size: pd.DataFrame(np.full((size, 6), np.nan), columns=COLUMNS)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                gen.generate_cell(path, config, 20, DAG, bad)
            with h5py.File(gen.partial_path(path), "r") as handle:
                self.assertFalse(handle["completed"][0])

    def test_main_reuses_existing_run_and_refuses_changed_settings(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            graph = nx.from_numpy_array(DAG, create_using=nx.DiGraph)
            graph = nx.relabel_nodes(graph, dict(enumerate(COLUMNS)))
            frame = pd.DataFrame(np.ones((50, 6)), columns=COLUMNS)
            argv = ["--source-root", str(root / "source"), "--output-root", str(root / "output"),
                    "--sample-sizes", "20", "--datasets-per-cell", "2"]
            def config_for(args, *unused):
                return {**protocol(count=args.datasets_per_cell, sizes=args.sample_sizes), "sampling_seed": args.seed}
            with mock.patch.object(gen, "load_reference", return_value=(frame, graph, COLUMNS, {})), \
                 mock.patch.object(gen, "generation_protocol", side_effect=config_for), \
                 mock.patch.object(gen, "load_or_fit", return_value=FakeCell()) as factory:
                gen.main(argv)
                gen.main(argv)
                gen.main(argv + ["--validate-only"])
                self.assertEqual(factory.call_count, 1)
                with self.assertRaisesRegex(ValueError, "different protocol"):
                    gen.main(argv + ["--seed", "43"])
            self.assertEqual(json.loads((root / "output/validation_report.json").read_text())["total_datasets"], 2)


if __name__ == "__main__":
    unittest.main()
