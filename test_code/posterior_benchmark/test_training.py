"""Exact distribution identities plus real CPU generation/train/resume/evaluation."""
import contextlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import torch

from test_code.posterior_benchmark import generate_training_data as gen
from test_code.posterior_benchmark import run_streaming as stream
from test_code.posterior_benchmark.data import (
    batch_plan, manifest_fingerprint, prepare_splits, read_batch, split_manifest,
    sample_training_observations,
)
from test_code.posterior_benchmark.engine import (
    evaluate_checkpoint, torch_load, run_key,
)
from test_code.posterior_benchmark.evaluate import evaluate_run
from test_code.posterior_benchmark.metrics import distribution_metrics
from test_code.posterior_benchmark.models import (
    METHODS, PAIRS, PERMUTATIONS, ModelConfig, PosteriorModel, PL,
    graph_log_probabilities, graph_nll, graph_tables, propagate_log_distribution,
)


def small_config(**kwargs):
    return ModelConfig(**{**dict(dim_model=16, dim_feedforward=32, nhead=2,
        encoder_layers=2, skeleton_layers=1, denoiser_layers=1, dropout=0.,
        reverse_steps=(0, 1, 2)), **kwargs})


class ProbabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.catalogue = gen.load_catalogue(Path(gen.__file__).with_name("all_25_sems.tex"))

    def test_graph_marginal_matches_all_48_skeleton_order_pairs_and_gradients(self):
        c = self.catalogue
        bits, compatible = graph_tables(c.graphs)
        logits = torch.tensor([[.4, -1.2, .8]], dtype=torch.float64, requires_grad=True)
        log_order = torch.log_softmax(torch.tensor([[.1, .3, -.2, .7, -1., .5]], dtype=torch.float64), -1)
        actual = graph_log_probabilities(logits, log_order,
                                         torch.tensor(bits), torch.tensor(compatible))
        expected = np.zeros(25)
        for mask in itertools.product((0, 1), repeat=3):
            weight = np.prod([float(torch.sigmoid(logits[0, i]).detach()) if bit else
                              1-float(torch.sigmoid(logits[0, i]).detach()) for i, bit in enumerate(mask)])
            for k, order in enumerate(PERMUTATIONS):
                pos = np.argsort(order)
                graph = np.zeros((3, 3), dtype=np.int8)
                for (left, right), bit in zip(PAIRS, mask):
                    if bit:
                        graph[(left, right) if pos[left] < pos[right] else (right, left)] = 1
                index = np.flatnonzero(np.all(c.graphs == graph, axis=(1, 2))).item()
                expected[index] += weight * float(log_order[0, k].exp())
        np.testing.assert_allclose(actual.exp().detach().numpy()[0], expected, atol=1e-14)
        self.assertAlmostEqual(float(actual.exp().sum()), 1., places=13)
        (-actual[0, 9]).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_compatible_orders_are_summed_not_averaged_and_prior_is_not_uniform_order(self):
        bits, compatible = graph_tables(self.catalogue.graphs)
        result = graph_log_probabilities(torch.zeros(1, 3),
            torch.full((1, 6), -math.log(6), dtype=torch.float64),
            torch.tensor(bits), torch.tensor(compatible)).exp()[0]
        self.assertAlmostEqual(float(result[0]), 1/8, places=12)  # empty graph: six orders
        self.assertAlmostEqual(float(result[9]), 1/48, places=12)  # complete graph: one order
        self.assertFalse(np.allclose(result.numpy(), np.full(25, 1/25)))

    def test_pl_six_probabilities_match_manual_formula(self):
        scores = torch.tensor([[.4, -1., .8]])
        permutations = torch.tensor(PERMUTATIONS)
        values = PL.log_prob(scores.unsqueeze(1), permutations.unsqueeze(0)).exp()[0]
        expected = []
        weights = np.exp(scores.numpy()[0])
        for a, b, c in PERMUTATIONS:
            expected.append(weights[a]/weights.sum() * weights[b]/(weights[b]+weights[c]))
        np.testing.assert_allclose(values.numpy(), expected, rtol=3e-7, atol=1e-8)

    def test_gpl_relative_transition_direction_and_exact_path_summation(self):
        torch.manual_seed(2)
        model = PosteriorModel("diffusion", small_config(), self.catalogue.graphs).eval()
        representation = torch.randn(1, 3, 16)
        k2 = model.transition_log_probs(representation, 2)
        k1 = model.transition_log_probs(representation, 1)
        np.testing.assert_allclose(k2.exp().sum(-1).detach().numpy(), 1., atol=4e-7)
        current = model.permutations.unsqueeze(0)
        scores = model.denoiser(current, torch.tensor([2]), representation)
        for i, source in enumerate(PERMUTATIONS):
            for j, target in enumerate(PERMUTATIONS):
                relative = torch.tensor([source.index(node) for node in target]).unsqueeze(0)
                manual = PL.log_prob_generalized_PL(scores[:, i], relative)
                self.assertAlmostEqual(float(k2[0, i, j]), float(manual), places=6)
        expected = torch.einsum('i,ij,jk->k', torch.ones(6, dtype=torch.float64)/6,
                                k2[0].exp(), k1[0].exp())
        actual = model._conditional_order_log_probs(representation).exp()[0]
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
        self.assertTrue(torch.allclose(propagate_log_distribution(
            torch.full((1,6), -math.log(6), dtype=torch.float64), k2).exp(),
            k2.exp().mean(1), atol=1e-14))

    def test_same_encoder_initialization_same_skeleton_and_raw_input(self):
        models = {}
        for method in METHODS:
            torch.manual_seed(9)
            models[method] = PosteriorModel(method, small_config(), self.catalogue.graphs)
        for name, value in models["categorical"].encoder.state_dict().items():
            for model in models.values():
                torch.testing.assert_close(value, model.encoder.state_dict()[name], rtol=0, atol=0)
        for name, value in models["pl"].skeleton.state_dict().items():
            torch.testing.assert_close(value, models["diffusion"].skeleton.state_dict()[name], rtol=0, atol=0)
        data = torch.randn(2, 13, 3) * 6 + 9
        observed = []
        hook = models["categorical"].encoder.embedder.register_forward_pre_hook(
            lambda module, args: observed.append(args[0].detach().clone()))
        models["categorical"](data)
        hook.remove()
        torch.testing.assert_close(observed[0].squeeze(-1), data, rtol=0, atol=0)

    def test_all_models_and_optional_priority_mixture_normalize_and_backpropagate(self):
        for priority_samples in (0, 2):
            for method in METHODS:
                model = PosteriorModel(method, small_config(priority_samples=priority_samples), self.catalogue.graphs)
                result = model(torch.randn(2, 9, 3))
                torch.testing.assert_close(result.exp().sum(-1), torch.ones(2, dtype=torch.float64), atol=1e-13, rtol=0)
                graph_nll(result, torch.tensor([0, 16])).backward()
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.encoder.parameters()
                                       if p.grad is not None), 0.)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                model.eval()
                representation = model.encode(torch.randn(2, 9, 3))
                if method != "categorical" and priority_samples:
                    pieces = [model._conditional_order_log_probs(representation, u.expand(2, -1)).exp()
                              for u in model.priority_bank]
                    torch.testing.assert_close(model.order_log_probs(representation).exp(),
                                               torch.stack(pieces).mean(0), atol=1e-13, rtol=0)

    def test_requested_n100_and_n2000_are_both_used_in_full(self):
        for method in METHODS:
            model = PosteriorModel(method, small_config(), self.catalogue.graphs).eval()
            with torch.inference_mode():
                for n in (100, 2000):
                    data = torch.randn(1, n, 3)
                    seen = []
                    hook = model.encoder.embedder.register_forward_pre_hook(
                        lambda module, args: seen.append(args[0].shape[1]))
                    log_p = model(data)
                    hook.remove()
                    self.assertEqual(seen, [n])
                    self.assertTrue(torch.isfinite(log_p).all())
                    self.assertAlmostEqual(float(log_p.exp().sum()), 1., places=12)

    def test_metrics_exact_nonidentifiable_posterior_has_zero_kl_but_positive_nll(self):
        c = self.catalogue
        sigma = c.mec_covariances[3]
        _, log_p, _, _ = gen.exact_posterior(c, (2000 * sigma)[None], 2000)
        values = distribution_metrics(log_p, log_p, np.array([3]), c.mec_ids)
        for key in ("kl", "tv", "js", "mec_kl", "within_true_mec_tv"):
            self.assertLess(values[key][0], 1e-13)
        self.assertAlmostEqual(values["graph_nll"][0], math.log(3), places=10)
        self.assertAlmostEqual(values["pred_mec_mass"][0], 1., places=10)

    def test_kl_direction_and_tv_checked_independently(self):
        p = np.arange(1, 26, dtype=float)
        q = p[::-1].copy()
        p /= p.sum()
        q /= q.sum()
        values = distribution_metrics(np.log(p)[None], np.log(q)[None], np.array([2]), self.catalogue.mec_ids)
        self.assertAlmostEqual(values["kl"][0], np.sum(p*np.log(p/q)), places=13)
        self.assertAlmostEqual(values["tv"][0], np.abs(p-q).sum()/2, places=13)
        with self.assertRaises(ValueError):
            distribution_metrics(np.log(p)[None], (np.log(q)+1)[None], np.array([2]), self.catalogue.mec_ids)



class LegacyCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.catalogue = gen.load_catalogue(Path(gen.__file__).with_name("all_25_sems.tex"))

    def test_denoising_objective_matches_original_functions_and_never_calls_graph_dp(self):
        from ml2_meta_causal_discovery.models.topo_order_diffusion import (
            CausalTopoOrderDiffusion as Old, CausalPriorityTopoOrderDiffusion as Priority,
            BakStyleSkeletonMixin,
        )
        for k in (0, 2):
            model = PosteriorModel("diffusion", small_config(priority_samples=k), self.catalogue.graphs)
            data, classes = torch.randn(2, 11, 3), torch.tensor([3, 16])
            rng = torch.get_rng_state()
            with mock.patch.object(model, "order_log_probs", side_effect=AssertionError("No graph DP in denoising training")):
                actual = model.training_loss(data, classes)
            torch.set_rng_state(rng)
            representation = model.encode(data)
            graph = model.graphs[classes]
            skeleton = BakStyleSkeletonMixin._skeleton_loss_per_batch(
                model, model.skeleton(representation), graph, mask=None).mean()
            if k:
                u = torch.rand((2,3))
                order = Priority._batched_priority_kahn_topological_orders(graph,u,torch.ones_like(u,dtype=torch.bool))
                ordered = Old._reorder_node_repr(representation,order)
                order_loss = Priority._training_losses_per_batch_with_priority(
                    model,ordered,Priority._reorder_priority(u,order)).mean()
            else:
                order = Old._sample_batch_topological_orders(model,graph)
                order_loss = Old._training_losses_per_batch(
                    model,Old._reorder_node_repr(representation,order)).mean()
            torch.testing.assert_close(actual["order_loss"],order_loss,atol=0,rtol=0)
            torch.testing.assert_close(actual["skeleton_loss"],skeleton,atol=0,rtol=0)
            torch.testing.assert_close(actual["loss"],.2*order_loss+skeleton,atol=0,rtol=0)
            actual["loss"].backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_legacy_observation_sampler_is_called_without_a_shared_plan(self):
        from ml2_meta_causal_discovery.utils.datautils import shared_batch_observation_indices as original
        batch = {"data":np.arange(4*2000*3,dtype=float).reshape(4,2000,3)}
        kwargs = dict(sample_size_min=20,sample_size_max=2001,sample_size_sampling="hybrid",
                      sample_size_hybrid_log_probability=.5,sample_size_hybrid_uniform_min=100)
        seen = []
        for seed in range(12):
            np.random.seed(seed)
            expected = original([(x,None) for x in batch["data"]],**kwargs)
            np.random.seed(seed)
            actual = sample_training_observations(batch,kwargs)
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual(len(set(actual.tolist())),len(actual))
            seen.append(len(actual))
        self.assertGreater(len(set(seen)),3)

    def test_pl_and_categorical_training_remain_exact_graph_nll(self):
        for method in ("categorical","pl"):
            model=PosteriorModel(method,small_config(),self.catalogue.graphs).eval()
            data,classes=torch.randn(2,11,3),torch.tensor([3,16])
            expected=graph_nll(model(data),classes)
            actual=model.training_loss(data,classes)
            torch.testing.assert_close(actual["loss"],expected,atol=0,rtol=0)


class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def arguments(self, root, chunks=1, method="diffusion", seed=0, extra=()):
        return ["--run-root",str(root),"--method",method,"--training-seed",str(seed),
                "--chunks",str(chunks),"--sample-sizes","5,7","--train-per-chunk","50",
                "--val-per-size","25","--test-per-size","25",
                "--stored-observations","32","--sample-size-min","4","--sample-size-max","33",
                "--sample-size-hybrid-uniform-min","10","--datasets-per-file","25",
                "--generation-batch-size","13","--batch-size","8","--dim-model","16",
                "--dim-feedforward","32","--nhead","2","--encoder-layers","2",
                "--skeleton-layers","1","--denoiser-layers","1","--dropout",".1",
                "--reverse-steps","0,1,2","--warmup-steps","2","--checkpoint-every","3",
                "--progress-every","100","--device","cpu","--cpu-threads","1",
                "--min-free-gb","0",*extra]

    def run_stream(self, arguments):
        with contextlib.redirect_stdout(io.StringIO()):
            return stream.main(arguments)

    def compare_models(self,a,b,method="diffusion"):
        left=torch_load(a/"checkpoints"/run_key(method,0)/"latest.pt")
        right=torch_load(b/"checkpoints"/run_key(method,0)/"latest.pt")
        for key in ("datasets_seen","global_step","data_history","sampling_history","observations_seen"):
            self.assertEqual(left[key],right[key],key)
        for name,value in left["model"].items():
            torch.testing.assert_close(value,right["model"][name],atol=0,rtol=0,msg=name)

    def test_three_independent_trainers_delete_only_their_data_and_share_only_test_X(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            roots=[base/method for method in METHODS]
            fingerprints=[]
            for method,root in zip(METHODS,roots):
                args=self.arguments(root,chunks=2,method=method,extra=("--delete-trained-chunks",))
                self.assertEqual(self.run_stream(args),0)
                state=json.loads((root/"run_state.json").read_text())
                self.assertEqual(state["status"],"training_complete")
                self.assertEqual(len(state["completed_chunks"]),2)
                self.assertEqual(state["runs_per_chunk"],1)
                self.assertFalse(any((root/"data/chunks").iterdir()))
                self.assertFalse((root/"data/fixed").exists())
                self.assertFalse((root/"evaluation").exists())
                self.assertFalse((root/"validation").exists())
                ck=torch_load(root/"checkpoints"/run_key(method,0)/"latest.pt")
                self.assertEqual(ck["datasets_seen"],100)
                fingerprints.append(ck["data_history"][0])
                self.assertTrue((root/"data_protocols/chunk_00001.json").is_file())
                self.assertEqual(self.run_stream([*args,"--resume"]),0)
            self.assertEqual(len(set(fingerprints)),3)
            out=base/"comparison"
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate_run(roots,output_root=out,device="cpu",batch_size=8)
            ids=[]
            for method in METHODS:
                with np.load(out/run_key(method,0)/"predictions/n7.npz") as p:
                    np.testing.assert_allclose(p["pred_posterior"].sum(-1),1,atol=1e-13)
                    self.assertEqual(p["pred_posterior"].shape,(25,25))
                    ids.append(p["dataset_seed"].tolist())
            self.assertTrue(all(value==ids[0] for value in ids))
            self.assertTrue((roots[0]/"data/fixed/test").exists())
            self.assertFalse((roots[1]/"data/fixed").exists())
            import csv
            with (out/"raw_metrics.csv").open() as f:
                raw=list(csv.DictReader(f))
            with (out/"summary_by_samples.csv").open() as f:
                summaries=list(csv.DictReader(f))
            self.assertEqual(len(raw),150)
            for row in summaries:
                values=[float(r["kl"]) for r in raw if r["method"]==row["method"] and r["sample_size"]==row["sample_size"]]
                self.assertAlmostEqual(float(row["mean_kl"]),np.mean(values),places=12)
                self.assertAlmostEqual(float(row["std_kl"]),np.std(values,ddof=1),places=12)

    def test_extend_preserves_optimizer_rng_and_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temporary:
            a,b=Path(temporary)/"a",Path(temporary)/"b"
            self.run_stream(self.arguments(a,chunks=2))
            self.run_stream(self.arguments(b))
            self.run_stream(self.arguments(b,chunks=2,extra=("--resume","--extend-target")))
            self.compare_models(a,b)

    def test_ctrl_c_resumes_identical_observation_draws_and_denoising_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            a,b=Path(temporary)/"a",Path(temporary)/"b"
            self.run_stream(self.arguments(a,extra=("--delete-trained-chunks",)))
            step=torch.optim.AdamW.step
            fired=[]
            def interrupted(optimizer,*args,**kwargs):
                result=step(optimizer,*args,**kwargs)
                if not fired:
                    fired.append(True)
                    os.kill(os.getpid(),signal.SIGINT)
                return result
            args=self.arguments(b,extra=("--delete-trained-chunks",))
            with mock.patch.object(torch.optim.AdamW,"step",interrupted):
                self.assertEqual(self.run_stream(args),130)
            self.assertTrue((b/"data/chunks/chunk_00001/train").is_dir())
            self.assertEqual(self.run_stream([*args,"--resume"]),0)
            self.compare_models(a,b)

    def test_failure_does_not_delete_data_and_protocol_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"run"
            args=self.arguments(root,extra=("--delete-trained-chunks",))
            with mock.patch.object(stream,"train_chunk",side_effect=RuntimeError("training failure")):
                with self.assertRaisesRegex(RuntimeError,"training failure"):
                    self.run_stream(args)
            self.assertTrue((root/"data/chunks/chunk_00001/train").is_dir())
            self.assertEqual(self.run_stream([*args,"--resume"]),0)
            with self.assertRaisesRegex(ValueError,"protocol changed"):
                self.run_stream([*args,"--resume","--learning-rate",".003"])

    def test_store_2000_then_subsample_and_never_standardize_or_evaluate(self):
        import test_code.posterior_benchmark.engine as engine
        original_sampler=engine.sample_training_observations
        original_encode=PosteriorModel.encode
        expected=[]
        def capture(batch,sampling):
            indices=original_sampler(batch,sampling)
            expected.append(batch["data"][:,indices,:].astype(np.float32))
            return indices
        checked=[]
        def check_encode(model,data):
            np.testing.assert_array_equal(data.detach().cpu().numpy(),expected[len(checked)])
            checked.append(True)
            return original_encode(model,data)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"run"
            args=self.arguments(root,method="categorical",extra=(
                "--train-per-chunk","25","--stored-observations","2000",
                "--sample-size-min","17","--sample-size-max","17",
                "--sample-size-hybrid-uniform-min","17"))
            with mock.patch.object(engine,"sample_training_observations",capture), \
                 mock.patch.object(PosteriorModel,"encode",check_encode), \
                 mock.patch.object(engine,"evaluate_checkpoint",side_effect=AssertionError("No automatic evaluation")):
                self.run_stream(args)
            self.assertTrue(checked)
            self.assertTrue(all(x.shape[1]==17 for x in expected))
            file=root/"data/chunks/chunk_00001/train/n2000/d3_shard_00000.hdf5"
            with h5py.File(file,"r") as h:
                self.assertEqual(h["data"].shape,(25,2000,3))
                self.assertFalse(np.allclose(h["data"][0].mean(0),0,atol=1e-12))
                self.assertEqual(h.attrs["input_normalization"],"none")
            ck=torch_load(root/"checkpoints/categorical_seed0/latest.pt")
            self.assertEqual(ck["observations_seen"],25*17)
            self.assertFalse((root/"data/fixed").exists())

    def test_training_seeds_are_independent_and_aggregate_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            roots=[base/f"s{seed}" for seed in (0,1)]
            for seed,root in enumerate(roots):
                self.run_stream(self.arguments(root,method="categorical",seed=seed))
            with contextlib.redirect_stdout(io.StringIO()):
                result=evaluate_run(roots,output_root=base/"combined",device="cpu",batch_size=8)
            self.assertEqual(len(result),4)
            self.assertEqual({r["training_seed"] for r in result},{0,1})

    def test_append_after_manual_test_writes_a_new_fixed_chunk_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"run"
            self.run_stream(self.arguments(root,method="pl"))
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate_run(root,device="cpu",batch_size=8)
            old=root/"evaluation/test/chunk_00001/raw_metrics.csv"
            digest=gen.file_hash(old)
            self.run_stream(self.arguments(root,chunks=2,method="pl",extra=("--resume","--extend-target")))
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate_run(root,device="cpu",batch_size=8)
            self.assertEqual(gen.file_hash(old),digest)
            self.assertTrue((root/"evaluation/test/chunk_00002/raw_metrics.csv").exists())

    def test_startup_interruption_can_resume_without_existing_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"run"
            original=gen.atomic_json
            def interrupt(path,value):
                original(path,value)
                if Path(path).name=="run_config.json":
                    raise KeyboardInterrupt
            with mock.patch.object(gen,"atomic_json",interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_stream(self.arguments(root))
            self.assertEqual(self.run_stream(self.arguments(root,extra=("--resume",))),0)

    def test_dry_run_and_deletion_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"run"
            self.run_stream(self.arguments(root,extra=("--dry-run",)))
            self.assertFalse(root.exists())
            unrelated=Path(temporary)/"unrelated"
            unrelated.mkdir()
            with self.assertRaisesRegex(ValueError,"Refusing deletion"):
                stream.retire_chunk(root,unrelated,"bad")
            self.assertTrue(unrelated.exists())

    def test_raw_X_and_posterior_corruption_are_rejected(self):
        catalogue=gen.load_catalogue(Path(gen.__file__).with_name("all_25_sems.tex"))
        with tempfile.TemporaryDirectory() as temporary,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temporary)/"data"
            rows=prepare_splits(root,catalogue,[7],dict(train=25,val=25,test=25),("test",),321,25,8)["test"]
            split_manifest(root,"test",verify=True)
            with h5py.File(rows[0]["path"],"r+") as h:
                h["data"][0,0,0]+=10
            with self.assertRaises(AssertionError):
                split_manifest(root,"test",verify=True)

    def test_prediction_cache_corruption_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"run"
            self.run_stream(self.arguments(root,method="categorical"))
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate_run(root,device="cpu",batch_size=8)
            path=root/"evaluation/test/chunk_00001/categorical_seed0/predictions/n7.npz"
            path.write_bytes(b"damaged fixture")
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,"changed evaluation"):
                evaluate_run(root,device="cpu",batch_size=8)


if __name__ == "__main__":
    unittest.main()
