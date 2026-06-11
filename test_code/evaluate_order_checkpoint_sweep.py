#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep classifier/bak checkpoints and save order metrics/distributions.

For the direct 4-node order classifier, this script records the exact softmax
probability of each of the D! orders. For bak/GS models, it samples many hard
permutations and records the empirical frequency of each order.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    encode_input,
    load_model,
    parse_csv_list,
    parse_int_list,
    remove_diag,
    sample_bak_orders,
    summarize_orders,
)
from test_code.evaluate_4var_order_classifier_vs_bak import load_classifier  # noqa: E402
from test_code.train_4var_order_classifier import (  # noqa: E402
    all_permutation_orders,
    valid_order_mask,
)


COMMON_METRICS = [
    "mean_topo_valid_rate",
    "mean_mode_order_valid",
    "mean_num_unique_orders",
    "mean_mode_order_fraction",
    "mean_avg_num_violations",
    "mean_avg_violation_rate",
]
CLASSIFIER_EXTRA_METRICS = ["mean_valid_mass"]
PLOT_METRICS = COMMON_METRICS + CLASSIFIER_EXTRA_METRICS


def parse_checkpoints(
    explicit: str,
    start: int,
    end: int,
    prefix: str,
    suffix: str,
) -> List[str]:
    if explicit:
        checkpoints = [x.strip() for x in explicit.split(",") if x.strip()]
    else:
        checkpoints = [f"{prefix}{i}{suffix}" for i in range(start, end + 1)]
    if not checkpoints:
        raise ValueError("No checkpoints selected.")
    return checkpoints


def checkpoint_index(checkpoint: str) -> int:
    match = re.search(r"(\d+)(?=\.pt$|$)", checkpoint)
    return int(match.group(1)) if match else -1


def order_to_string(order: Sequence[int]) -> str:
    return "".join(str(int(x)) for x in order)


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


class DistributionAccumulator:
    def __init__(self, group_cols: Sequence[str], order_strings: Sequence[str]) -> None:
        self.group_cols = list(group_cols)
        self.order_strings = list(order_strings)
        self.sums: Dict[Tuple[Any, ...], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.counts: Dict[Tuple[Any, ...], int] = defaultdict(int)

    def add(self, group: Dict[str, Any], distribution: Dict[str, float]) -> None:
        key = tuple(group[c] for c in self.group_cols)
        self.counts[key] += 1
        for order in self.order_strings:
            self.sums[key][order] += float(distribution.get(order, 0.0))

    def to_frame(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for key in sorted(self.counts):
            n = self.counts[key]
            base = {col: val for col, val in zip(self.group_cols, key)}
            base["n_datasets"] = n
            for order in self.order_strings:
                row = dict(base)
                row["order"] = order
                row["mean_probability"] = self.sums[key][order] / max(n, 1)
                rows.append(row)
        return pd.DataFrame(rows)


def classifier_distribution_and_metrics(
    *,
    model: Any,
    data: np.ndarray,
    true_dag: np.ndarray,
    orders_24: torch.Tensor,
    order_strings: Sequence[str],
    num_order_samples: int,
    device: str,
    standardize: bool,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    inputs = encode_input(data, device=device, standardize=standardize)
    target = torch.tensor(true_dag, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        scores = model(inputs, orders=orders_24, mask=None)
        probs = torch.softmax(scores.float(), dim=-1)
        valid = valid_order_mask(target, orders_24)
        sample_idx = torch.multinomial(probs, num_samples=num_order_samples, replacement=True)
        sampled_orders = orders_24[sample_idx[0]]

        valid_mass = (probs * valid.float()).sum(dim=-1)[0]

    probs_np = probs[0].detach().cpu().numpy().astype(float)
    distribution = {order: float(probs_np[i]) for i, order in enumerate(order_strings)}
    metrics = {
        "valid_mass": float(valid_mass.item()),
    }
    return sampled_orders.detach().cpu().numpy().astype(int), metrics, distribution


def bak_distribution_from_samples(orders: np.ndarray, order_strings: Sequence[str]) -> Dict[str, float]:
    counts = {order: 0 for order in order_strings}
    for order in orders:
        key = order_to_string(order)
        if key in counts:
            counts[key] += 1
    denom = max(int(orders.shape[0]), 1)
    return {order: counts[order] / denom for order in order_strings}


def add_per_dataset_distribution_rows(
    rows: List[Dict[str, Any]],
    *,
    group: Dict[str, Any],
    dataset_id: str,
    distribution: Dict[str, float],
    order_strings: Sequence[str],
) -> None:
    for order in order_strings:
        row = dict(group)
        row["dataset_id"] = dataset_id
        row["order"] = order
        row["probability"] = float(distribution.get(order, 0.0))
        rows.append(row)


def evaluate(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    methods = set(parse_csv_list(args.methods))
    valid_methods = {"classifier", "bak"}
    unknown_methods = methods - valid_methods
    if unknown_methods:
        raise ValueError(f"Unknown methods: {sorted(unknown_methods)}. Use any of: {sorted(valid_methods)}")

    sample_sizes = parse_int_list(args.sample_sizes)
    files_by_n = build_file_index(args, sample_sizes)
    orders_24 = all_permutation_orders(args.num_nodes, torch.device(device))
    order_strings = [order_to_string(order.detach().cpu().tolist()) for order in orders_24]

    group_cols = ["benchmark", "generator", "graph_id", "graph_name", "num_samples", "method", "checkpoint", "checkpoint_index", "variant"]
    aggregate_group_cols = ["benchmark", "num_samples", "method", "checkpoint", "checkpoint_index", "variant"]
    detailed = RunningSummary(group_cols)
    aggregate = RunningSummary(aggregate_group_cols)
    distribution = DistributionAccumulator(group_cols, order_strings)
    per_dataset_distribution_rows: List[Dict[str, Any]] = []

    model_specs: List[Dict[str, Any]] = []
    if "classifier" in methods:
        classifier_checkpoints = parse_checkpoints(
            args.classifier_checkpoints,
            args.classifier_checkpoint_start,
            args.classifier_checkpoint_end,
            args.classifier_checkpoint_prefix,
            args.classifier_checkpoint_suffix,
        )
        for checkpoint in classifier_checkpoints:
            model_specs.append(
                {
                    "kind": "classifier",
                    "method": args.classifier_method_name,
                    "run_name": args.classifier_run_name,
                    "checkpoint": checkpoint,
                }
            )
    if "bak" in methods:
        bak_checkpoints = parse_checkpoints(
            args.bak_checkpoints,
            args.bak_checkpoint_start,
            args.bak_checkpoint_end,
            args.bak_checkpoint_prefix,
            args.bak_checkpoint_suffix,
        )
        for checkpoint in bak_checkpoints:
            model_specs.append(
                {
                    "kind": "bak",
                    "method": args.bak_method_name,
                    "run_name": args.bak_run_name,
                    "checkpoint": checkpoint,
                }
            )

    print("=" * 100)
    print("Order checkpoint sweep")
    print(f"methods:           {sorted(methods)}")
    print(f"benchmark_kind:    {args.benchmark_kind}")
    print(f"benchmark_root:    {Path(args.benchmark_root).expanduser().resolve()}")
    print(f"sample_sizes:      {sample_sizes}")
    print(f"num_order_samples: {args.num_order_samples}")
    print(f"device:            {device}")
    print("=" * 100)

    classifier_root = Path(args.classifier_results_dir).expanduser().resolve()
    models_root = Path(args.models_root).expanduser().resolve()
    bak_path = Path(args.bak_model_file).expanduser().resolve()
    start = time.time()

    for spec_idx, spec in enumerate(model_specs, start=1):
        kind = spec["kind"]
        checkpoint = spec["checkpoint"]
        ckpt_idx = checkpoint_index(checkpoint)
        variant = f"{spec['run_name']}_{checkpoint}_orders{args.num_order_samples}"

        print("=" * 100)
        print(f"[{spec_idx}/{len(model_specs)}] Loading {kind}: {spec['run_name']}/{checkpoint}")
        if kind == "classifier":
            run_dir = classifier_root / spec["run_name"]
            checkpoint_path = run_dir / checkpoint
            if not checkpoint_path.exists():
                if args.skip_missing_checkpoints:
                    print(f"[SKIP] missing classifier checkpoint: {checkpoint_path}")
                    continue
                raise FileNotFoundError(f"Missing classifier checkpoint: {checkpoint_path}")
            model, _, loaded_path = load_classifier(run_dir=run_dir, checkpoint=checkpoint, device=device)
        else:
            checkpoint_path = models_root / spec["run_name"] / checkpoint
            if not checkpoint_path.exists():
                if args.skip_missing_checkpoints:
                    print(f"[SKIP] missing bak checkpoint: {checkpoint_path}")
                    continue
                raise FileNotFoundError(f"Missing bak checkpoint: {checkpoint_path}")
            model, _, loaded_path = load_model(
                models_root=models_root,
                run_name=spec["run_name"],
                checkpoint=checkpoint,
                source="bak",
                device=device,
                bak_path=bak_path,
            )
        print(f"loaded: {loaded_path}")

        for n, files in files_by_n.items():
            print("-" * 100)
            print(f"{kind} checkpoint={checkpoint} n={n}: files={len(files)}")
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
                        dataset_id = f"{h5_path.name}:{data_idx}"
                        seen_for_n += 1

                        group = {
                            "benchmark": meta["benchmark"],
                            "generator": meta["generator"],
                            "graph_id": meta["graph_id"],
                            "graph_name": meta["graph_name"],
                            "num_samples": n,
                            "method": spec["method"],
                            "checkpoint": checkpoint,
                            "checkpoint_index": ckpt_idx,
                            "variant": variant,
                        }
                        aggregate_group = {
                            "benchmark": meta["benchmark"],
                            "num_samples": n,
                            "method": spec["method"],
                            "checkpoint": checkpoint,
                            "checkpoint_index": ckpt_idx,
                            "variant": variant,
                        }

                        try:
                            if kind == "classifier":
                                sampled_orders, exact_metrics, order_dist = classifier_distribution_and_metrics(
                                    model=model,
                                    data=data,
                                    true_dag=true_dag,
                                    orders_24=orders_24,
                                    order_strings=order_strings,
                                    num_order_samples=args.num_order_samples,
                                    device=device,
                                    standardize=args.standardize,
                                )
                                metrics = summarize_orders(true_dag, sampled_orders)
                                metrics.update(exact_metrics)
                            else:
                                sampled_orders = sample_bak_orders(
                                    model=model,
                                    data=data,
                                    num_order_samples=args.num_order_samples,
                                    device=device,
                                    standardize=args.standardize,
                                )
                                metrics = summarize_orders(true_dag, sampled_orders)
                                order_dist = bak_distribution_from_samples(sampled_orders, order_strings)

                            detailed.add_ok(group, metrics)
                            aggregate.add_ok(aggregate_group, metrics)
                            distribution.add(group, order_dist)
                            if args.save_per_dataset_distributions:
                                add_per_dataset_distribution_rows(
                                    per_dataset_distribution_rows,
                                    group=group,
                                    dataset_id=dataset_id,
                                    distribution=order_dist,
                                    order_strings=order_strings,
                                )
                        except Exception as e:
                            detailed.add_error(group)
                            aggregate.add_error(aggregate_group)
                            if args.print_errors:
                                print(f"    [ERROR] {kind} checkpoint={checkpoint} n={n} idx={data_idx}: {repr(e)}")
                    if args.max_datasets_per_n is not None and seen_for_n >= args.max_datasets_per_n:
                        break

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("=" * 100)
    print(f"Finished. elapsed={time.time() - start:.1f}s")
    per_dataset_frame = pd.DataFrame(per_dataset_distribution_rows)
    return aggregate.to_frame(), detailed.to_frame(), distribution.to_frame(), per_dataset_frame


def save_metric_plots(aggregate: pd.DataFrame, plots_dir: Path, prefix: str) -> List[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable; skip plots: {repr(e)}")
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = [
        c for c in PLOT_METRICS
        if c in aggregate.columns and pd.api.types.is_numeric_dtype(aggregate[c])
    ]
    written: List[Path] = []

    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        plotted = False
        for method in sorted(aggregate["method"].dropna().unique()):
            for n in sorted(aggregate["num_samples"].dropna().unique()):
                sub = aggregate[(aggregate["method"] == method) & (aggregate["num_samples"] == n)].sort_values("checkpoint_index")
                sub = sub.dropna(subset=[metric])
                if sub.empty:
                    continue
                ax.plot(
                    sub["checkpoint_index"],
                    sub[metric],
                    marker="o",
                    linewidth=1.6,
                    label=f"{method}, n={int(n)}",
                )
                plotted = True

        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("checkpoint index")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        out_path = plots_dir / f"{prefix}_{metric}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        written.append(out_path)
    return written


def save_order_distribution_grid_plots(distributions: pd.DataFrame, plots_dir: Path, prefix: str) -> List[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable; skip order distribution plots: {repr(e)}")
        return []

    if distributions.empty:
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    group_cols = [
        "benchmark",
        "generator",
        "graph_id",
        "graph_name",
        "num_samples",
        "method",
    ]
    available_group_cols = [c for c in group_cols if c in distributions.columns]

    for group_values, group_df in distributions.groupby(available_group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group = dict(zip(available_group_cols, group_values))
        orders = sorted(group_df["order"].unique())
        n_rows, n_cols = 4, 6
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18.0, 10.0), sharex=True, sharey=True)
        axes_flat = axes.reshape(-1)

        for ax, order in zip(axes_flat, orders):
            sub = group_df[group_df["order"] == order].sort_values("checkpoint_index")
            ax.plot(sub["checkpoint_index"], sub["mean_probability"], marker="o", linewidth=1.4)
            ax.set_title(order, fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.set_ylim(-0.02, 1.02)

        for ax in axes_flat[len(orders):]:
            ax.axis("off")

        method = str(group.get("method", "method"))
        benchmark = str(group.get("benchmark", "benchmark"))
        generator = str(group.get("generator", "generator"))
        graph_id = group.get("graph_id", "graph")
        num_samples = group.get("num_samples", "n")
        fig.suptitle(
            f"{method} order distribution: {benchmark}, {generator}, graph={graph_id}, n={num_samples}",
            fontsize=13,
        )
        fig.supxlabel("checkpoint index")
        fig.supylabel("mean probability / sample frequency")
        fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.95])

        safe_method = re.sub(r"[^A-Za-z0-9_.-]+", "_", method)
        safe_generator = re.sub(r"[^A-Za-z0-9_.-]+", "_", generator)
        out_path = plots_dir / (
            f"{prefix}_order_grid_{safe_method}_{benchmark}_{safe_generator}"
            f"_graph{graph_id}_n{num_samples}.png"
        )
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        written.append(out_path)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep classifier/bak checkpoints and save order distributions.")
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
    parser.add_argument("--methods", type=str, default="classifier,bak")

    parser.add_argument("--classifier_results_dir", type=str, default="result/order_classifier_4var")
    parser.add_argument("--classifier_run_name", type=str, default="order_classifier_4var_d128_bs16_fixedmask")
    parser.add_argument("--classifier_method_name", type=str, default="OrderClassifier")
    parser.add_argument("--classifier_checkpoints", type=str, default="")
    parser.add_argument("--classifier_checkpoint_start", type=int, default=0)
    parser.add_argument("--classifier_checkpoint_end", type=int, default=13)
    parser.add_argument("--classifier_checkpoint_prefix", type=str, default="model_")
    parser.add_argument("--classifier_checkpoint_suffix", type=str, default=".pt")

    parser.add_argument("--models_root", type=str, default="ml2_meta_causal_discovery/experiments/causal_classification/models")
    parser.add_argument("--bak_run_name", type=str, default="gp_4var_prob_bakL_100k")
    parser.add_argument("--bak_method_name", type=str, default="CSNP-bakL")
    parser.add_argument("--bak_model_file", type=str, default="ml2_meta_causal_discovery/models/causaltransformernp.py.mask_version.bak")
    parser.add_argument("--bak_checkpoints", type=str, default="")
    parser.add_argument("--bak_checkpoint_start", type=int, default=1)
    parser.add_argument("--bak_checkpoint_end", type=int, default=8)
    parser.add_argument("--bak_checkpoint_prefix", type=str, default="model_")
    parser.add_argument("--bak_checkpoint_suffix", type=str, default=".pt")

    parser.add_argument("--results_dir", type=str, default="result/order_checkpoint_sweep")
    parser.add_argument("--summary_prefix", type=str, default="order_checkpoint_sweep")
    parser.add_argument("--save_per_dataset_distributions", action="store_true")
    parser.add_argument("--max_files_per_n", type=int, default=None)
    parser.add_argument("--max_datasets_per_file", type=int, default=None)
    parser.add_argument("--max_datasets_per_n", type=int, default=None)
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--skip_missing_checkpoints", action="store_true")
    parser.add_argument("--print_errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / f"{args.summary_prefix}_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    aggregate, detailed, distributions, per_dataset_distributions = evaluate(args)
    keep_metric_cols = ["mean_" + m for m in [
        "topo_valid_rate",
        "mode_order_valid",
        "num_unique_orders",
        "mode_order_fraction",
        "avg_num_violations",
        "avg_violation_rate",
        "valid_mass",
    ]]
    base_cols = [c for c in aggregate.columns if not c.startswith("mean_")]
    aggregate = aggregate[base_cols + [c for c in keep_metric_cols if c in aggregate.columns]]
    base_cols = [c for c in detailed.columns if not c.startswith("mean_")]
    detailed = detailed[base_cols + [c for c in keep_metric_cols if c in detailed.columns]]

    aggregate_path = results_dir / f"{args.summary_prefix}_aggregate.csv"
    detailed_path = results_dir / f"{args.summary_prefix}_detailed.csv"
    distributions_path = results_dir / f"{args.summary_prefix}_order_distributions.csv"

    aggregate.to_csv(aggregate_path, index=False)
    detailed.to_csv(detailed_path, index=False)
    distributions.to_csv(distributions_path, index=False)
    print(f"Wrote aggregate summary:       {aggregate_path}")
    print(f"Wrote detailed summary:        {detailed_path}")
    print(f"Wrote order distributions:     {distributions_path}")

    if args.save_per_dataset_distributions:
        per_dataset_path = results_dir / f"{args.summary_prefix}_per_dataset_order_distributions.csv"
        per_dataset_distributions.to_csv(per_dataset_path, index=False)
        print(f"Wrote per-dataset distributions: {per_dataset_path}")

    plot_paths = save_metric_plots(aggregate, plots_dir, args.summary_prefix)
    if plot_paths:
        print(f"Wrote {len(plot_paths)} plots under: {plots_dir}")
    order_plot_paths = save_order_distribution_grid_plots(distributions, plots_dir, args.summary_prefix)
    if order_plot_paths:
        print(f"Wrote {len(order_plot_paths)} order distribution plots under: {plots_dir}")


if __name__ == "__main__":
    main()
