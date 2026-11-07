#!/usr/bin/env python3
"""Train ONE independent posterior model: generate n=2000, subsample, save, retire.

This is a separate posterior experiment. It never invokes or changes the old
classifier trainer, point-estimate inference, or any existing benchmark run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from test_code.posterior_benchmark import generate_training_data as gen
from test_code.posterior_benchmark.data import (
    derived_seed, exclusive_lock, manifest_fingerprint, prepare_splits,
)
from test_code.posterior_benchmark.engine import (
    device_for, run_key, train_chunk, torch_load,
)
from test_code.posterior_benchmark.models import METHODS, ModelConfig


def integers(value):
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("Use comma-separated integers") from None
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Values must be nonempty and distinct")
    return values


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--sem-file", type=Path, default=Path(__file__).with_name("all_25_sems.tex"))
    p.add_argument("--sample-sizes", type=integers, default=integers("20,50,100,300,500,1000,2000"))
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--training-seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=20261108)
    p.add_argument("--evaluation-seed", type=int, default=20261109,
                   help="Common fixed-test seed; independent of training method/seed.")
    p.add_argument("--chunks", type=int, default=10)
    p.add_argument("--train-per-chunk", type=int, default=70000,
                   help="Total fresh datasets per chunk; NOT multiplied by evaluation sample sizes.")
    p.add_argument("--stored-observations", type=int, default=2000)
    p.add_argument("--sample-size-min", type=int, default=20)
    p.add_argument("--sample-size-max", type=int, default=2001)
    p.add_argument("--sample-size-sampling", choices=("uniform", "log_uniform", "hybrid"), default="hybrid")
    p.add_argument("--sample-size-hybrid-log-probability", type=float, default=.5)
    p.add_argument("--sample-size-hybrid-uniform-min", type=int, default=100)
    p.add_argument("--val-per-size", type=int, default=2500)
    p.add_argument("--test-per-size", type=int, default=5000)
    p.add_argument("--datasets-per-file", type=int, default=5000)
    p.add_argument("--generation-batch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=3750)
    p.add_argument("--gradient-clip", type=float, default=1.)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--diffusion-loss", choices=("denoising", "graph_nll"), default="denoising")
    p.add_argument("--order-loss-weight", type=float, default=.2)
    p.add_argument("--skeleton-loss-weight", type=float, default=1.)
    p.add_argument("--dim-model", type=int, default=256)
    p.add_argument("--dim-feedforward", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--encoder-layers", type=int, default=4)
    p.add_argument("--skeleton-layers", type=int, default=2)
    p.add_argument("--denoiser-layers", type=int, default=7)
    p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--reverse-steps", type=integers, default=[0, 4, 7, 11])
    p.add_argument("--priority-samples", type=int, default=0,
                   help="0: no priority, exact enumeration. >0: shared fixed uniform-u MC bank for PL/GPL.")
    p.add_argument("--priority-seed", type=int, default=20261107)
    p.add_argument("--device", default="auto")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--min-free-gb", type=float, default=5.)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--extend-target", action="store_true")
    p.add_argument("--delete-trained-chunks", action="store_true")
    p.add_argument("--no-final-test", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    args.run_root = args.run_root.expanduser().resolve()
    args.sem_file = args.sem_file.expanduser().resolve()
    for value in (args.train_per_chunk, args.val_per_size, args.test_per_size, args.datasets_per_file):
        if value < 25 or value % 25:
            p.error("All split/shard counts must be positive multiples of 25")
    if (min(args.sample_sizes) < 2 or args.training_seed < 0 or args.data_seed < 0 or args.evaluation_seed < 0
        or min(args.chunks, args.batch_size, args.generation_batch_size,
               args.checkpoint_every, args.progress_every, args.cpu_threads) < 1):
        p.error("Invalid negative seed, sample size or nonpositive size/count")
    if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0
        or not math.isfinite(args.weight_decay) or args.weight_decay < 0
        or not math.isfinite(args.gradient_clip) or args.gradient_clip <= 0
        or args.warmup_steps < 0 or not math.isfinite(args.min_free_gb) or args.min_free_gb < 0):
        p.error("Invalid optimizer or free-space setting")
    if (args.stored_observations < 2 or not 2 <= args.sample_size_min <= args.sample_size_max
        or args.sample_size_max > args.stored_observations + 1
        or args.sample_size_min > args.stored_observations
        or not args.sample_size_min <= args.sample_size_hybrid_uniform_min <= args.sample_size_max
        or not 0 <= args.sample_size_hybrid_log_probability <= 1
        or (args.sample_size_sampling == "hybrid" and args.sample_size_hybrid_log_probability < 1
            and args.sample_size_hybrid_uniform_min >= args.sample_size_max
            and args.sample_size_min != args.sample_size_max)):
        p.error("Invalid stored row count or legacy observation sampling bounds")
    if (any(not math.isfinite(v) or v < 0 for v in (args.order_loss_weight, args.skeleton_loss_weight))
        or args.order_loss_weight + args.skeleton_loss_weight == 0):
        p.error("Diffusion loss weights must be nonnegative and not both zero")
    args.sample_sizes.sort()
    args.data_seed_base = args.data_seed
    args.data_seed = derived_seed(args.data_seed_base, f"independent/{args.method}/seed-{args.training_seed}")
    return args


def model_config(args):
    config = ModelConfig(
        dim_model=args.dim_model, dim_feedforward=args.dim_feedforward, nhead=args.nhead,
        encoder_layers=args.encoder_layers, skeleton_layers=args.skeleton_layers,
        denoiser_layers=args.denoiser_layers, dropout=args.dropout,
        reverse_steps=tuple(args.reverse_steps), priority_samples=args.priority_samples,
        priority_seed=args.priority_seed,
    )
    config.validate()
    return config


def make_protocol(args, catalogue):
    files = ["test_code/posterior_benchmark/" + name for name in
             ("models.py", "engine.py", "data.py", "metrics.py", "generate_training_data.py", "run_streaming.py", "evaluate.py")]
    files += ["ml2_meta_causal_discovery/models/causaltransformercomponents.py",
              "ml2_meta_causal_discovery/models/topo_order_diffusion.py",
              "ml2_meta_causal_discovery/utils/datautils.py",
              "ml2_meta_causal_discovery/utils/topological_orders.py",
              "SymmetricDiffusers/PL_distribution.py", "SymmetricDiffusers/utils.py",
              "SymmetricDiffusers/models.py"]
    train_names = ("batch_size", "learning_rate", "weight_decay", "warmup_steps",
                   "gradient_clip", "checkpoint_every", "progress_every", "diffusion_loss",
                   "order_loss_weight", "skeleton_loss_weight")
    sampling_names = ("sample_size_min", "sample_size_max", "sample_size_sampling",
                      "sample_size_hybrid_log_probability", "sample_size_hybrid_uniform_min")
    return {"format": "raw-X-posterior-independent-v3", "sem_sha256": catalogue.source_sha256,
            "sample_sizes": args.sample_sizes, "methods": [args.method],
            "training_seeds": [args.training_seed], "data_seed": args.data_seed,
            "data_seed_base": args.data_seed_base, "evaluation_seed": args.evaluation_seed,
            "train_datasets_per_chunk": args.train_per_chunk,
            "stored_observations": args.stored_observations,
            "observation_sampling": {name: getattr(args, name) for name in sampling_names},
            "evaluation_counts_per_size": {"val": args.val_per_size, "test": args.test_per_size},
            "evaluation_datasets_per_file": 5000,
            "datasets_per_file": args.datasets_per_file,
            "model": model_config(args).document(),
            "training": {key: getattr(args, key) for key in train_names},
            "input": "raw unstandardized X; fixed A/B/C; independent legacy minibatch row subsampling",
            "objective": {"categorical": "graph_nll", "pl": "graph_nll", "diffusion": args.diffusion_loss},
            "automatic_evaluation": False,
            "priority": ("none; exact order/path/graph enumeration" if not args.priority_samples else
                         "uniform-u model; fixed MC integration bank at evaluation; exact conditional graph summation"),
            "torch_version": str(torch.__version__), "numpy_version": np.__version__,
            "source_sha256": {name: gen.file_hash(ROOT / name) for name in files}}


def check_space(root, args):
    # Includes every field in the existing generator, plus an HDF5 overhead margin.
    estimated = args.train_per_chunk * (24 * args.stored_observations + 1700)
    available = shutil.disk_usage(root).free
    required = int(estimated * 1.15 + args.min_free_gb * 2**30)
    if available < required:
        raise OSError(f"Insufficient free disk: {available/2**30:.1f} GiB, "
                      f"need ~{required/2**30:.1f} GiB including reserve")


def retire_chunk(root, chunk_dir, protocol_hash):
    """Only an owned, fully committed generated chunk can be removed."""
    root, chunk_dir = Path(root).resolve(), Path(chunk_dir)
    expected_parent = root / "data" / "chunks"
    if (chunk_dir.is_symlink() or chunk_dir.parent.resolve() != expected_parent
        or not chunk_dir.name.startswith("chunk_") or chunk_dir.resolve() != expected_parent / chunk_dir.name):
        raise ValueError("Refusing deletion outside the exact owned chunk directory")
    state = json.loads((root / "run_state.json").read_text())
    committed = [r for r in state["completed_chunks"] if r["directory"] == chunk_dir.name]
    owner = json.loads((chunk_dir / "stream_owner.json").read_text())
    if (len(committed) != 1 or owner != {"run_root": str(root), "protocol_hash": protocol_hash}
        or len(committed[0]["runs"]) != state["runs_per_chunk"]):
        raise ValueError("Chunk is not owned and committed for every model/seed")
    for entry in committed[0]["runs"]:
        checkpoint = root / entry["checkpoint"]
        if not checkpoint.is_file() or gen.file_hash(checkpoint) != entry["checkpoint_sha256"]:
            raise ValueError("Checkpoint missing/changed; retaining training data")
    if any(path.is_symlink() for path in chunk_dir.rglob("*")):
        raise ValueError("Unexpected symlink in chunk; refusing recursive deletion")
    shutil.rmtree(chunk_dir)
    print(f"[REMOVED GENERATED CHUNK] {chunk_dir}; seeds/protocol/checkpoints retained for regeneration", flush=True)


def execute(args):
    catalogue = gen.load_catalogue(args.sem_file)
    equivalence = gen.check_sem_equivalence(catalogue)
    protocol = make_protocol(args, catalogue)
    protocol_hash = hashlib.sha256(gen.canonical(protocol).encode()).hexdigest()
    total = args.train_per_chunk * args.chunks
    if args.dry_run:
        print(json.dumps({"run_root": str(args.run_root), "protocol": protocol,
                          "chunks": args.chunks, "training_datasets_per_model": total,
                          "delete_trained_chunks": args.delete_trained_chunks}, indent=2))
        return
    root = args.run_root
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / ".stream.lock"):
        config_path, state_path = root / "run_config.json", root / "run_state.json"
        if config_path.exists():
            if not args.resume:
                raise ValueError("Existing run: explicitly use --resume; nothing overwritten")
            if json.loads(config_path.read_text()) != protocol:
                raise ValueError("Run/data/model/source protocol changed; use a new run-root")
            if state_path.exists():
                state = json.loads(state_path.read_text())
            elif any((root / "checkpoints").glob("**/*.pt")):
                raise ValueError("Run state missing but checkpoints exist; refusing to infer training history")
            else:
                # A stop immediately after the initial config commit has not
                # trained anything. Recreate only the empty progress state.
                state = {"protocol_hash": protocol_hash, "target_chunks": args.chunks,
                         "runs_per_chunk": 1,
                         "completed_chunks": [], "status": "running"}
            if state["protocol_hash"] != protocol_hash:
                raise ValueError("Run-state protocol mismatch")
            if args.chunks != state["target_chunks"]:
                if args.chunks < state["target_chunks"] or not args.extend_target:
                    raise ValueError("To increase chunk target use --extend-target; cannot shrink it")
                state["target_chunks"] = args.chunks
        else:
            initial_files = {".stream.lock", "catalogue.json", "all_25_sems.tex", "sem_equivalence.json"}
            if any(path.name not in initial_files for path in root.iterdir()):
                raise ValueError("Refusing unrelated nonempty run-root")
            if (root / "all_25_sems.tex").exists() and gen.file_hash(root / "all_25_sems.tex") != catalogue.source_sha256:
                raise ValueError("Partial initialization uses a different SEM prior")
            gen.atomic_json(root / "catalogue.json", gen.catalogue_document(catalogue))
            gen.atomic_bytes(root / "all_25_sems.tex", catalogue.source.read_bytes())
            gen.atomic_json(root / "sem_equivalence.json", equivalence)
            gen.atomic_json(config_path, protocol)
            state = {"protocol_hash": protocol_hash, "target_chunks": args.chunks,
                     "runs_per_chunk": 1,
                     "completed_chunks": [], "status": "running"}
            gen.atomic_json(state_path, state)
        state["status"] = "running"
        gen.atomic_json(state_path, state)
        device = device_for(args.device)
        torch.set_num_threads(args.cpu_threads)
        # Also repair harmless missing initial snapshots after a startup stop.
        if not (root / "all_25_sems.tex").exists():
            gen.atomic_bytes(root / "all_25_sems.tex", catalogue.source.read_bytes())
        if not (root / "catalogue.json").exists():
            gen.atomic_json(root / "catalogue.json", gen.catalogue_document(catalogue))
        print(f"device={device}; stored_n={args.stored_observations}; input_normalization=none; "
              f"diffusion_loss={args.diffusion_loss}; no automatic validation/test; "
              f"method={args.method}; seed={args.training_seed}; total/model={total}", flush=True)
        for committed in state["completed_chunks"]:
            old = root / "data" / "chunks" / committed["directory"]
            if args.delete_trained_chunks and old.exists():
                retire_chunk(root, old, protocol_hash)
        for index in range(len(state["completed_chunks"]) + 1, args.chunks + 1):
            directory = root / "data" / "chunks" / f"chunk_{index:05d}"
            if not (directory / "generation_config.json").exists():
                check_space(root, args)
            counts = {"train": args.train_per_chunk, "val": args.val_per_size, "test": args.test_per_size}
            manifests = prepare_splits(
                directory, catalogue, [args.stored_observations], counts, ("train",),
                derived_seed(args.data_seed, f"training-chunk-{index}"), args.datasets_per_file,
                args.generation_batch_size)
            gen.atomic_json(directory / "stream_owner.json", {"run_root": str(root), "protocol_hash": protocol_hash})
            manifest = manifests["train"]
            checkpoint, completed = train_chunk(
                root, protocol, protocol_hash, catalogue, args.method, args.training_seed, index, manifest, device)
            if not completed:
                state["status"] = "interrupted"
                gen.atomic_json(state_path, state)
                print("INTERRUPTED SAFELY: training data retained; rerun with --resume", flush=True)
                return 130
            checkpoint_state = torch_load(checkpoint)
            trace = checkpoint_state["sampling_trace_sha256"]
            histogram = checkpoint_state["minibatch_n_histogram"]
            del checkpoint_state
            runs = [{"key": run_key(args.method, args.training_seed),
                     "checkpoint": str(checkpoint.relative_to(root)),
                     "checkpoint_sha256": gen.file_hash(checkpoint), "sampling_trace_sha256": trace}]
            metadata_dir = root / "data_protocols"
            metadata_dir.mkdir(exist_ok=True)
            gen.atomic_json(metadata_dir / f"chunk_{index:05d}.json", {
                "generation_config": json.loads((directory / "generation_config.json").read_text()),
                "manifest": [{**row, "path": str(Path(row["path"]).relative_to(directory))} for row in manifest],
                "observation_sampling": protocol["observation_sampling"],
                "sampling_trace_sha256": trace, "minibatch_n_histogram": histogram})
            state["completed_chunks"].append({"index": index, "directory": directory.name,
                "datasets_per_model": sum(row["count"] for row in manifest),
                "data_fingerprint": manifest_fingerprint(manifest), "sampling_trace_sha256": trace,
                "runs": runs})
            gen.atomic_json(state_path, state)
            if args.delete_trained_chunks:
                retire_chunk(root, directory, protocol_hash)
        state["status"] = "training_complete"
        gen.atomic_json(state_path, state)
        print(f"POSTERIOR TRAINING COMPLETE: chunks={args.chunks} datasets/model={total} "
              f"run_root={root}; evaluation not run", flush=True)
    return 0


def main(argv=None):
    return execute(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
