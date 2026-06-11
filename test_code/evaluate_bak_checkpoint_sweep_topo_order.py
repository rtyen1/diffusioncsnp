#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate bak topological-order quality across multiple checkpoints.

This script reuses the order-sampling logic from evaluate_topo_order_4var_summary.py.
For each checkpoint it samples Q/order samples from the bak probabilistic decoder,
checks whether the sampled root-to-leaf orders are valid topological orders of the
true DAG, writes CSV summaries, and plots one curve per metric.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_topo_order_4var_summary import (  # noqa: E402
    RunningSummary,
    discover_h5_files,
    load_model,
    parse_int_list,
    remove_diag,
    sample_bak_orders,
    summarize_orders,
)


def parse_checkpoints(args: argparse.Namespace) -> List[str]:
    if args.checkpoints:
        checkpoints = [x.strip() for x in args.checkpoints.split(",") if x.strip()]
    else:
        checkpoints = [
            f"{args.checkpoint_prefix}{i}{args.checkpoint_suffix}"
            for i in range(args.checkpoint_start, args.checkpoint_end + 1)
        ]
    if not checkpoints:
        raise ValueError("No checkpoints selected.")
    return checkpoints


def checkpoint_index(checkpoint: str) -> int:
    match = re.search(r"(\d+)(?=\.pt$|$)", checkpoint)
    return int(match.group(1)) if match else -1


def checkpoint_variant(run_name: str, checkpoint: str, num_order_samples: int) -> str:
    return f"{run_name}_{checkpoint}_orders{num_order_samples}"


def build_file_index(args: argparse.Namespace, sample_sizes: Sequence[int]) -> Dict[int, List[Tuple[Path, Dict[str, Any]]]]:
    files_by_n: Dict[int, List[Tuple[Path, Dict[str, Any]]]] = {}
    for n in sample_sizes:
        files = discover_h5_files(args, n)
        if args.max_files_per_n is not None:
            files = files[: args.max_files_per_n]
        if not files:
            if args.skip_missing:
                print(f"[SKIP] no h5 files for n={n}")
                continue
            raise FileNotFoundError(f"No h5 files found for n={n}.")
        files_by_n[n] = files
    return files_by_n


def evaluate_one_checkpoint(
    *,
    args: argparse.Namespace,
    model: Any,
    checkpoint: str,
    files_by_n: Dict[int, List[Tuple[Path, Dict[str, Any]]]],
    aggregate: RunningSummary,
    detailed: RunningSummary,
    device: str,
) -> None:
    ckpt_idx = checkpoint_index(checkpoint)
    variant = checkpoint_variant(args.bak_run_name, checkpoint, args.num_order_samples)

    for n, files in files_by_n.items():
        print("-" * 100)
        print(f"checkpoint={checkpoint} n={n}: files={len(files)}")
        seen_for_n = 0
        for file_idx, (h5_path, meta) in enumerate(files):
            with h5py.File(h5_path, "r") as f:
                data_arr = f["data"]
                label_arr = f["label"]
                file_count = int(data_arr.shape[0])
                limit = file_count if args.max_datasets_per_file is None else min(file_count, args.max_datasets_per_file)
                print(
                    f"  [{file_idx + 1}/{len(files)}] graph={meta['graph_id']} "
                    f"generator={meta['generator']} {h5_path.name} datasets={limit}/{file_count}"
                )
                for data_idx in range(limit):
                    if args.max_datasets_per_n is not None and seen_for_n >= args.max_datasets_per_n:
                        break
                    data = np.asarray(data_arr[data_idx], dtype=np.float32)
                    true_dag = remove_diag(np.asarray(label_arr[data_idx], dtype=int))
                    seen_for_n += 1

                    detailed_group = {
                        "benchmark": meta["benchmark"],
                        "generator": meta["generator"],
                        "graph_id": meta["graph_id"],
                        "graph_name": meta["graph_name"],
                        "num_samples": n,
                        "method": args.bak_method_name,
                        "checkpoint": checkpoint,
                        "checkpoint_index": ckpt_idx,
                        "variant": variant,
                    }
                    aggregate_group = {
                        "benchmark": meta["benchmark"],
                        "num_samples": n,
                        "method": args.bak_method_name,
                        "checkpoint": checkpoint,
                        "checkpoint_index": ckpt_idx,
                        "variant": variant,
                    }
                    try:
                        orders = sample_bak_orders(
                            model=model,
                            data=data,
                            num_order_samples=args.num_order_samples,
                            device=device,
                            standardize=args.standardize,
                        )
                        metrics = summarize_orders(true_dag, orders)
                        detailed.add_ok(detailed_group, metrics)
                        aggregate.add_ok(aggregate_group, metrics)
                    except Exception as e:
                        detailed.add_error(detailed_group)
                        aggregate.add_error(aggregate_group)
                        if args.print_errors:
                            print(f"    [ERROR] checkpoint={checkpoint} n={n} idx={data_idx}: {repr(e)}")
                if args.max_datasets_per_n is not None and seen_for_n >= args.max_datasets_per_n:
                    break


def save_metric_plots(aggregate: pd.DataFrame, plots_dir: Path, prefix: str) -> List[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable; skip plots: {repr(e)}")
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = [c for c in aggregate.columns if c.startswith("mean_")]
    metric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(aggregate[c])]
    written: List[Path] = []

    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        plotted = False
        for n in sorted(aggregate["num_samples"].dropna().unique()):
            sub = aggregate[aggregate["num_samples"] == n].sort_values("checkpoint_index")
            sub = sub.dropna(subset=[metric])
            if sub.empty:
                continue
            ax.plot(
                sub["checkpoint_index"],
                sub[metric],
                marker="o",
                linewidth=1.8,
                label=f"n={int(n)}",
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel("checkpoint index")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        out_path = plots_dir / f"{prefix}_{metric}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        written.append(out_path)
    return written


def evaluate(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    sample_sizes = parse_int_list(args.sample_sizes)
    checkpoints = parse_checkpoints(args)
    models_root = Path(args.models_root).expanduser().resolve()
    bak_path = Path(args.bak_model_file).expanduser().resolve()
    files_by_n = build_file_index(args, sample_sizes)

    detailed = RunningSummary(
        [
            "benchmark",
            "generator",
            "graph_id",
            "graph_name",
            "num_samples",
            "method",
            "checkpoint",
            "checkpoint_index",
            "variant",
        ]
    )
    aggregate = RunningSummary(
        ["benchmark", "num_samples", "method", "checkpoint", "checkpoint_index", "variant"]
    )

    print("=" * 100)
    print("Bak checkpoint sweep: topology-order evaluation")
    print(f"benchmark_kind:    {args.benchmark_kind}")
    print(f"benchmark_root:    {Path(args.benchmark_root).expanduser().resolve()}")
    print(f"models_root:       {models_root}")
    print(f"run_name:          {args.bak_run_name}")
    print(f"checkpoints:       {checkpoints}")
    print(f"sample_sizes:      {sample_sizes}")
    print(f"num_order_samples: {args.num_order_samples}")
    print(f"device:            {device}")
    print("=" * 100)

    start = time.time()
    for ckpt_i, checkpoint in enumerate(checkpoints, start=1):
        model_path = models_root / args.bak_run_name / checkpoint
        if not model_path.exists():
            if args.skip_missing_checkpoints:
                print(f"[SKIP] missing checkpoint: {model_path}")
                continue
            raise FileNotFoundError(f"Missing checkpoint: {model_path}")

        print("=" * 100)
        print(f"[{ckpt_i}/{len(checkpoints)}] Loading {args.bak_run_name}/{checkpoint}")
        model, _, loaded_path = load_model(
            models_root=models_root,
            run_name=args.bak_run_name,
            checkpoint=checkpoint,
            source="bak",
            device=device,
            bak_path=bak_path,
        )
        print(f"loaded: {loaded_path}")
        evaluate_one_checkpoint(
            args=args,
            model=model,
            checkpoint=checkpoint,
            files_by_n=files_by_n,
            aggregate=aggregate,
            detailed=detailed,
            device=device,
        )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("=" * 100)
    print(f"Finished. elapsed={time.time() - start:.1f}s")
    return aggregate.to_frame(), detailed.to_frame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep bak checkpoints and plot topology-order metrics.")
    parser.add_argument("--benchmark_root", type=str, default="benchmark_data_4var")
    parser.add_argument("--benchmark_kind", type=str, default="random_gp", choices=["random_gp", "fixed_graph"])
    parser.add_argument("--distribution", type=str, default="csnp_gp_4var_ERL0U1")
    parser.add_argument("--fixed_generators", type=str, default="csnp_gp,rff_gaussian")
    parser.add_argument("--graph_ids", type=str, default="all")
    parser.add_argument("--num_nodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_sizes", type=str, default="5,20,50,100,300,1000,3000")
    parser.add_argument("--num_order_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--standardize", action="store_true", default=True)
    parser.add_argument("--no_standardize", action="store_false", dest="standardize")

    parser.add_argument("--models_root", type=str, default="ml2_meta_causal_discovery/experiments/causal_classification/models")
    parser.add_argument("--bak_run_name", type=str, required=True)
    parser.add_argument("--bak_method_name", type=str, default="CSNP-bakL")
    parser.add_argument("--bak_model_file", type=str, default="ml2_meta_causal_discovery/models/causaltransformernp.py.mask_version.bak")
    parser.add_argument("--checkpoints", type=str, default="", help="Comma-separated checkpoint names. Overrides start/end.")
    parser.add_argument("--checkpoint_start", type=int, default=1)
    parser.add_argument("--checkpoint_end", type=int, default=8)
    parser.add_argument("--checkpoint_prefix", type=str, default="model_")
    parser.add_argument("--checkpoint_suffix", type=str, default=".pt")
    parser.add_argument("--skip_missing_checkpoints", action="store_true")

    parser.add_argument("--results_dir", type=str, default="result/bak_checkpoint_sweep_topo_order")
    parser.add_argument("--summary_prefix", type=str, default="bak_checkpoint_sweep")
    parser.add_argument("--max_files_per_n", type=int, default=None)
    parser.add_argument("--max_datasets_per_file", type=int, default=None)
    parser.add_argument("--max_datasets_per_n", type=int, default=None)
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--print_errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / f"{args.summary_prefix}_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    aggregate, detailed = evaluate(args)

    aggregate_path = results_dir / f"{args.summary_prefix}_aggregate.csv"
    detailed_path = results_dir / f"{args.summary_prefix}_detailed.csv"
    aggregate.to_csv(aggregate_path, index=False)
    detailed.to_csv(detailed_path, index=False)
    print(f"Wrote aggregate summary: {aggregate_path}")
    print(f"Wrote detailed summary:  {detailed_path}")

    plot_paths = save_metric_plots(aggregate, plots_dir, args.summary_prefix)
    if plot_paths:
        print(f"Wrote {len(plot_paths)} plots under: {plots_dir}")


if __name__ == "__main__":
    main()
