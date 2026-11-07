"""Exact likelihood, SEM equivalence, shared-prior labels and I/O checks."""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

from test_code.posterior_benchmark import generate_training_data as gen


class PosteriorDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogue = gen.load_catalogue(Path(gen.__file__).with_name("all_25_sems.tex"))

    def arguments(self, root, batch=7, extra=()):
        return ["--output-root", str(root), "--sample-sizes", "20,50",
                "--train-per-size", "50", "--val-per-size", "25", "--test-per-size", "25",
                "--datasets-per-file", "25", "--batch-size", str(batch), "--seed", "71", *extra]

    def test_catalogue_all_graphs_mecs_positive_covariances_and_direction(self):
        c = self.catalogue
        self.assertEqual(c.graphs.shape, (25, 3, 3))
        self.assertEqual(c.mec_covariances.shape, (11, 3, 3))
        np.testing.assert_array_equal(np.bincount(c.mec_ids)[1:], [1,2,2,3,2,3,1,6,1,3,1])
        self.assertTrue((np.linalg.eigvalsh(c.mec_covariances) > 0).all())
        self.assertEqual(c.coefficients[1, 2, 1], .5)  # DAG #2 C -> B
        self.assertEqual(c.graphs[1, 2, 1], 1)
        self.assertEqual(c.graphs[1, 1, 2], 0)
        for i, graph in enumerate(c.graphs):
            self.assertFalse(np.tril(graph[np.ix_(c.orders[i], c.orders[i])]).any())
            covariance = c.sem_factors[i] @ c.sem_factors[i].T
            np.testing.assert_allclose(covariance, c.mec_covariances[int(c.mec_ids[i])-1], atol=1e-14)

    def test_shared_noise_sem_and_covariance_coupling(self):
        result = gen.check_sem_equivalence(self.catalogue)
        self.assertEqual(result["models_checked"], 25)
        self.assertLess(result["max_shared_noise_sem_matrix_error"], 1e-13)

    def test_joint_gaussian_likelihood_equals_product_of_sem_conditionals(self):
        c = self.catalogue
        data = np.random.default_rng(8).normal(size=(13, 3))
        for index in range(25):
            residuals = data - data @ c.coefficients[index]
            variances = c.noise_variances[index]
            sem_logpdf = -.5*np.sum(np.log(2*math.pi*variances) + residuals**2/variances)
            mec = int(c.mec_ids[index])-1
            gaussian_logpdf = -.5*(13*3*math.log(2*math.pi) + 13*c.log_determinants[mec]
                + np.einsum("ni,ij,nj->", data, c.precisions[mec], data))
            self.assertAlmostEqual(sem_logpdf, gaussian_logpdf, places=11)

    def test_monte_carlo_sem_and_covariance_sampling_have_the_declared_law(self):
        c = self.catalogue
        z = np.random.default_rng(19).standard_normal((100_000, 3))
        for index in (0, 3, 16, 23):
            sigma = c.mec_covariances[int(c.mec_ids[index])-1]
            for data in (gen.sample_sem(c, index, z), z @ c.cholesky[int(c.mec_ids[index])-1].T):
                np.testing.assert_allclose(data.mean(axis=0), 0, atol=.025)
                np.testing.assert_allclose(np.cov(data, rowvar=False, ddof=0), sigma, atol=.05)

    def test_zero_observations_gives_uniform_dag_not_uniform_mec_prior(self):
        p, lp, mec, evidence = gen.exact_posterior(self.catalogue, np.zeros((3, 3)), 0)
        np.testing.assert_allclose(p, np.full(25, 1/25), atol=1e-15)
        np.testing.assert_allclose(mec, np.array([1,2,2,3,2,3,1,6,1,3,1])/25, atol=1e-15)
        self.assertAlmostEqual(float(evidence), 0, places=14)
        self.assertTrue(np.isfinite(lp).all())

    def test_exact_posterior_matches_independent_sem_density_and_linear_softmax(self):
        c = self.catalogue
        data = np.random.default_rng(8).normal(size=(4, 13, 3)) + .3
        scatter, features = gen.sufficient_statistics(data)
        p, lp, _, _ = gen.exact_posterior(c, scatter, 13)
        log_joint = np.empty((4,25))
        for index in range(25):
            residuals = data - data @ c.coefficients[index]
            variances = c.noise_variances[index]
            log_joint[:,index] = -.5*np.sum(np.log(2*math.pi*variances) + residuals**2/variances,
                                           axis=(1,2)) + math.log(1/25)
        shifted = log_joint-log_joint.max(axis=-1, keepdims=True)
        expected = np.exp(shifted)/np.exp(shifted).sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(p, expected, rtol=1e-12, atol=1e-14)
        weights, bias = gen.analytic_softmax_parameters(c)
        logits = features @ weights.T + bias
        shifted = logits-logits.max(axis=-1,keepdims=True)
        np.testing.assert_allclose(p, np.exp(shifted)/np.exp(shifted).sum(axis=-1,keepdims=True), atol=1e-14)
        np.testing.assert_allclose(np.exp(lp), p, atol=1e-15)
        # The known mean is zero: do not replace X^T X by centered covariance.
        np.testing.assert_allclose(scatter, data.transpose(0,2,1) @ data, atol=1e-13)
        centered = data-data.mean(axis=1,keepdims=True)
        self.assertGreater(float(np.max(np.abs(scatter - centered.transpose(0,2,1) @ centered))), .1)

    def test_nonidentifiable_mec4_remains_three_way_uncertain(self):
        c = self.catalogue
        x = np.random.default_rng(4).normal(size=(4, 20, 3))
        p, _, mec, _ = gen.exact_posterior(c, gen.sufficient_statistics(x)[0], 20)
        for group in range(1,12):
            indices = np.flatnonzero(c.mec_ids == group)
            np.testing.assert_array_equal(p[:,indices], np.repeat(p[:,indices[:1]], len(indices), axis=1))
        p, _, mec, _ = gen.exact_posterior(c, 5000*c.mec_covariances[3], 5000)
        np.testing.assert_allclose(p[[3,5,13]], np.full(3, 1/3), atol=1e-10)
        self.assertGreater(mec[3], 1-1e-10)
        self.assertLess(p.max(), .334)

    def test_extreme_likelihoods_do_not_create_nan_or_lose_log_probabilities(self):
        p, lp, mec, evidence = gen.exact_posterior(self.catalogue, np.eye(3)*1e8, 100_000)
        self.assertTrue(np.isfinite(p).all() and np.isfinite(lp).all())
        self.assertTrue(np.isfinite(mec).all() and np.isfinite(evidence).all())
        self.assertAlmostEqual(float(p.sum()), 1., places=9)

    def test_parser_refuses_changed_or_inconsistent_sem(self):
        original = self.catalogue.source.read_text()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"bad.tex"
            path.write_text(original.replace("B &= 0.5 C", "B &= 0.7 C", 1))
            with self.assertRaisesRegex(ValueError, "covariance mismatch"):
                gen.load_catalogue(path)

    def test_seed_replay_and_graph_uniformity(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = gen.parse_args(self.arguments(Path(tmp)))
            config = gen.run_config(args, self.catalogue)
            jobs = gen.jobs_for(args)
            seeds = []
            for job in jobs:
                metadata = gen.expected_metadata(config, job, self.catalogue)
                np.testing.assert_array_equal(np.bincount(metadata["class_index"], minlength=25), np.ones(25,dtype=int))
                np.testing.assert_array_equal(metadata["graph_id"], metadata["class_index"]+1)
                np.testing.assert_array_equal(metadata["mec_id"], self.catalogue.mec_ids[metadata["class_index"]])
                seeds.extend(metadata["dataset_seed"].tolist())
            self.assertEqual(len(set(seeds)), len(seeds))

    def test_complete_files_reader_replay_batch_invariance_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            gen.main(self.arguments(a, batch=7))
            gen.main(self.arguments(b, batch=11))
            gen.main(self.arguments(a, batch=5, extra=["--validate-only"]))
            first_file = None
            for path in a.glob("*/n*/d3_shard_*.hdf5"):
                first_file = path
                with h5py.File(path) as left, h5py.File(b/path.relative_to(a)) as right:
                    for key in left:
                        np.testing.assert_array_equal(left[key][:], right[key][:])
                    seed = int(left["dataset_seed"][0])
                    cls = int(left["class_index"][0])
                    replay = gen.sample_batch(self.catalogue, np.array([cls]), np.array([seed],dtype=np.uint64), int(left.attrs["n"]))
                    np.testing.assert_array_equal(replay[0], left["data"][0])
                    self.assertEqual(left.attrs["input_normalization"], "none")
                    self.assertGreater(float(np.max(np.abs(left["data"][0].mean(axis=0)))), 1e-4)
            report = json.loads((a/"validation_report.json").read_text())
            self.assertEqual(report["total_datasets"], 200)
            self.assertEqual(report["status"], "COMPLETE")
            checksum = gen.file_hash(first_file)
            with self.assertRaisesRegex(ValueError, "protocol differs"):
                gen.main(self.arguments(a, extra=["--seed", "72"]))
            self.assertEqual(checksum, gen.file_hash(first_file))
            with mock.patch.object(gen, "sample_batch", side_effect=AssertionError("must not regenerate")):
                gen.main(self.arguments(a))

    def test_partial_resume_uses_original_seeds_and_keeps_finished_rows(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = gen.parse_args(self.arguments(Path(tmp)/"resume"))
            config = gen.run_config(args, self.catalogue)
            job = gen.jobs_for(args)[0]
            original = gen.sample_batch
            calls = 0
            def interrupt(*a, **kw):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("test interruption")
                return original(*a, **kw)
            with mock.patch.object(gen, "sample_batch", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    gen.write_shard(config, job, self.catalogue, 7)
            path = Path(job["path"])
            partial = path.with_name(f".{path.stem}.partial.hdf5")
            self.assertFalse(path.exists())
            with h5py.File(partial) as handle:
                self.assertEqual(handle["completed"][:].sum(), 7)
                saved = handle["data"][:7]
            gen.write_shard(config, job, self.catalogue, 5)
            clean = {**job, "path": str(Path(tmp)/"clean/file.hdf5")}
            gen.write_shard(config, clean, self.catalogue, 25)
            with h5py.File(path) as actual, h5py.File(clean["path"]) as reference:
                np.testing.assert_array_equal(saved, actual["data"][:7])
                for key in actual:
                    np.testing.assert_array_equal(actual[key][:], reference[key][:])

    def test_posterior_corruption_and_old_directory_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)/"output"
            gen.main(self.arguments(root))
            path = next(root.glob("train/n*/d3_shard_*.hdf5"))
            with h5py.File(path, "r+") as handle:
                handle["posterior"][0,0] += .1
            with self.assertRaises(AssertionError):
                gen.main(self.arguments(root, extra=["--validate-only"]))
            old = Path(tmp)/"old"
            old.mkdir()
            (old/"user_data.hdf5").touch()
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                gen.main(self.arguments(old))
            self.assertTrue((old/"user_data.hdf5").exists())

    def test_sem_sampling_option_preserves_parent_to_child_equations(self):
        classes = np.array([0,3,16,23], dtype=np.uint8)
        seeds = np.array([81,82,83,84],dtype=np.uint64)
        batch = gen.sample_batch(self.catalogue, classes, seeds, 50, "sem")
        for i, (cls, seed) in enumerate(zip(classes, seeds)):
            z = np.random.default_rng(int(seed)).standard_normal((50,3))
            np.testing.assert_array_equal(batch[i], gen.sample_sem(self.catalogue, int(cls), z))

    def test_counts_must_be_multiples_of_25(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                gen.parse_args(self.arguments(Path(tmp), extra=["--train-per-size", "51"]))


if __name__ == "__main__":
    unittest.main()
