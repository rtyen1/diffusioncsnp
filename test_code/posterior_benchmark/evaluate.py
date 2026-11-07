#!/usr/bin/env python3
"""Manually evaluate independent posterior models on ONE fixed shared test set."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from test_code.posterior_benchmark import generate_training_data as gen
from test_code.posterior_benchmark.data import (
    split_manifest, exclusive_lock, manifest_fingerprint, prepare_fixed_evaluation_data,
)
from test_code.posterior_benchmark.engine import device_for, evaluate_checkpoint, run_key, torch_load
from test_code.posterior_benchmark.metrics import write_csv


def evaluate_run(run_root, checkpoint="last_completed", split="test", output_root=None,
                 device="auto", batch_size=16, overwrite=False):
    """run_root may be one path or a list of independent run paths."""
    roots = list(run_root) if isinstance(run_root, (list, tuple)) else [run_root]
    roots = [Path(p).expanduser().resolve() for p in roots]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Expected distinct run roots")
    if len(roots) > 1 and output_root is None:
        raise ValueError("Provide --output-root for a comparison of independent runs")
    protocols, selections, checkpoint_map = [], [], {}
    for root in roots:
        protocol = json.loads((root / "run_config.json").read_text())
        if protocol["format"] != "raw-X-posterior-independent-v3":
            raise ValueError("Use a new independent v3 run; old runs/checkpoints are left unchanged")
        for relative, expected in protocol["source_sha256"].items():
            if gen.file_hash(ROOT / relative) != expected:
                raise ValueError(f"Source changed since training: {relative}; refusing incompatible evaluation")
        protocols.append(protocol)
        selection = checkpoint
        if selection in {"latest", "last_completed"}:
            state = json.loads((root / "run_state.json").read_text())
            if not state["completed_chunks"]:
                raise ValueError(f"No completed training chunk in {root}")
            selection = f"chunk_{state['completed_chunks'][-1]['index']:05d}"
        if not selection.startswith("chunk_") or not selection[6:].isdigit():
            raise ValueError("checkpoint must be last_completed or a fixed chunk_00001 archive")
        selections.append(selection)
        for seed in protocol["training_seeds"]:
            for method in protocol["methods"]:
                key = run_key(method, seed)
                if key in checkpoint_map:
                    raise ValueError(f"Duplicate method/seed in comparison: {key}")
                path = root / "checkpoints" / key / f"checkpoint_{selection}.pt"
                state = torch_load(path)
                if (state["method"] != method or state["training_seed"] != seed
                    or state["protocol"] != protocol or state["completed_chunks"] != state["active_chunk"]):
                    raise ValueError("Wrong model, protocol, or incomplete checkpoint")
                checkpoint_map[key] = {
                    "run_root": str(root), "path": str(path.relative_to(root)),
                    "sha256": gen.file_hash(path), "datasets_seen": state["datasets_seen"],
                    "data_history": state["data_history"], "sampling_history": state["sampling_history"],
                    "parameter_counts": state["parameter_counts"], "selected_chunk": selection,
                }
    if len({r["datasets_seen"] for r in checkpoint_map.values()}) != 1:
        raise ValueError("Comparison checkpoints have different training data budgets")
    first = protocols[0]
    for protocol in protocols[1:]:
        for field in ("sem_sha256", "sample_sizes", "evaluation_seed", "evaluation_datasets_per_file"):
            if protocol[field] != first[field]:
                raise ValueError(f"Different evaluation prior/protocol: {field}")
        if protocol["evaluation_counts_per_size"][split] != first["evaluation_counts_per_size"][split]:
            raise ValueError("Different evaluation dataset counts")
    # Training data and subsamples are intentionally independent. Only EVALUATION
    # must be paired, so every model reads the first root's exact same fixed X.
    catalogue = gen.load_catalogue(roots[0] / "all_25_sems.tex")
    prepare_fixed_evaluation_data(roots[0], first, catalogue, split)
    manifest = split_manifest(roots[0] / "data" / "fixed", split, verify=True)
    out = Path(output_root).expanduser().resolve() if output_root else roots[0] / "evaluation" / split / selections[0]
    for root in roots:
        if out == root or out in root.parents or any(
                out == p or p in out.parents for p in (root / "data", root / "checkpoints")):
            raise ValueError("Evaluation output must not replace a run/data/checkpoint root")
    out.mkdir(parents=True, exist_ok=True)
    summaries, raw_rows = [], []
    device = device_for(device)
    with exclusive_lock(out / ".evaluation.lock"):
        gen.atomic_json(out / "evaluation_config.json", {"status": "in_progress", "split": split})
        for key, record in checkpoint_map.items():
            path = Path(record["run_root"]) / record["path"]
            summaries.extend(evaluate_checkpoint(path, manifest, catalogue, out / key, device,
                                                batch_size, save_predictions=True, overwrite=overwrite))
            with (out / key / "raw_metrics.csv").open(newline="") as handle:
                raw_rows.extend(csv.DictReader(handle))
        expected_ids = None
        for key in checkpoint_map:
            method, seed_text = key.rsplit("_seed", 1)
            selected = [(int(r["sample_size"]), int(r["dataset_seed"])) for r in raw_rows
                        if r["method"] == method and int(r["training_seed"]) == int(seed_text)]
            ids = set(selected)
            if len(ids) != len(selected) or (expected_ids is not None and ids != expected_ids):
                raise ValueError("Models were not evaluated on identical test datasets")
            expected_ids = ids
        write_csv(out / "raw_metrics.csv", raw_rows)
        write_csv(out / "summary_by_samples.csv", summaries)
        across_seeds = []
        for method in sorted({r["method"] for r in summaries}):
            for n in first["sample_sizes"]:
                group = [r for r in summaries if r["method"] == method and r["sample_size"] == n]
                row = {"method": method, "sample_size": n, "training_seeds": len(group)}
                for metric in ("kl", "tv", "js", "graph_nll", "mec_kl", "pred_mec_mass", "within_true_mec_tv"):
                    values = [r[f"mean_{metric}"] for r in group]
                    row[f"mean_{metric}"] = float(np.mean(values))
                    row[f"std_across_training_seeds_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
                across_seeds.append(row)
        write_csv(out / "summary_across_training_seeds.csv", across_seeds)
        gen.atomic_json(out / "evaluation_config.json", {
            "status": "complete", "split": split, "requested_checkpoint": checkpoint,
            "checkpoints": checkpoint_map, "training_data_shared": False,
            "evaluation_data_shared": True, "data_fingerprint": manifest_fingerprint(manifest),
            "total_metric_rows": len(raw_rows), "posterior_order": "DAG #1..#25 in all_25_sems.tex",
            "summary_std": "sample SD across datasets at fixed method/seed/n; ddof=1",
            "seed_summary_std": "SD across training-seed means, not pooled dataset SD",
        })
    print(f"POSTERIOR EVALUATION COMPLETE: rows={len(raw_rows)} output={out}", flush=True)
    return summaries


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-root", type=Path)
    group.add_argument("--run-roots", type=Path, nargs="+")
    p.add_argument("--checkpoint", default="last_completed")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--output-root", type=Path)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if min(args.batch_size, args.cpu_threads) < 1:
        p.error("batch size and CPU thread count must be positive")
    torch.set_num_threads(args.cpu_threads)
    evaluate_run(args.run_roots or args.run_root, args.checkpoint, args.split, args.output_root,
                 args.device, args.batch_size, args.overwrite)


if __name__ == "__main__":
    main()
