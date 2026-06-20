#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate traditional order-producing causal discovery baselines on 4-node data.

The main metric checks whether the predicted root-to-leaf order is a valid
topological order of the true DAG: every true edge i -> j must satisfy
position(i) < position(j).

Supported methods:
  - direct_lingam: causal-learn DirectLiNGAM, uses model.causal_order_
  - ica_lingam: causal-learn ICALiNGAM, uses model.causal_order_
  - hillclimb_bic: pgmpy HillClimbSearch + Gaussian BIC, outputs a DAG then topological sort
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import pandas as pd

from evaluate_topo_order_4var_summary import (
    RunningSummary,
    discover_h5_files,
    is_valid_topological_order,
    maybe_standardize,
    parse_csv_list,
    parse_int_list,
    remove_diag,
)


def order_to_str(order: np.ndarray) -> str:
    return "".join(str(int(x)) for x in np.asarray(order, dtype=int).tolist())


def edge_precedence_accuracy(adj: np.ndarray, order: np.ndarray) -> float:
    adj = remove_diag(adj).astype(int)
    order = np.asarray(order, dtype=int)
    pos = np.empty(adj.shape[0], dtype=int)
    pos[order] = np.arange(len(order))
    edges = [(i, j) for i in range(adj.shape[0]) for j in range(adj.shape[1]) if adj[i, j] == 1]
    if not edges:
        return 1.0
    correct = sum(1 for i, j in edges if pos[i] < pos[j])
    return float(correct / len(edges))


def topological_sort_from_adj(adj: np.ndarray) -> np.ndarray:
    """Deterministic Kahn topological sort, picking smallest available node."""
    adj = remove_diag(adj).astype(int)
    num_nodes = adj.shape[0]
    indegree = adj.sum(axis=0).astype(int)
    available = [i for i in range(num_nodes) if indegree[i] == 0]
    order: List[int] = []

    while available:
        node = min(available)
        available.remove(node)
        order.append(node)
        for child in np.flatnonzero(adj[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                available.append(int(child))

    if len(order) != num_nodes:
        raise ValueError("Predicted graph is cyclic; cannot topologically sort it.")
    return np.asarray(order, dtype=int)


def run_direct_lingam(data: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    from causallearn.search.FCMBased.lingam import DirectLiNGAM

    x = maybe_standardize(data, standardize=args.standardize)
    model = DirectLiNGAM(
        random_state=args.seed,
        measure=args.direct_lingam_measure,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x)
    return np.asarray(model.causal_order_, dtype=int)


def run_ica_lingam(data: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    from causallearn.search.FCMBased.lingam import ICALiNGAM

    x = maybe_standardize(data, standardize=args.standardize)
    model = ICALiNGAM(
        random_state=args.seed,
        max_iter=args.ica_lingam_max_iter,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x)
    return np.asarray(model.causal_order_, dtype=int)


def bic_score_class():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            from pgmpy.estimators import BICGauss

            return BICGauss
        except ImportError:
            pass
        try:
            from pgmpy.estimators import BIC

            return BIC
        except ImportError:
            from pgmpy.estimators import BicScore

            return BicScore


def run_hillclimb_bic(data: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            from pgmpy.estimators import HillClimbSearch
    except ImportError as exc:
        raise ImportError(
            "hillclimb_bic requires pgmpy. Install it with `pip install pgmpy`."
        ) from exc

    x = maybe_standardize(data, standardize=args.standardize)
    columns = [f"X{i}" for i in range(x.shape[1])]
    frame = pd.DataFrame(x, columns=columns)
    score = bic_score_class()(frame)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        estimator = HillClimbSearch(frame)
        dag = estimator.estimate(scoring_method=score, show_progress=False)

    adj = np.zeros((x.shape[1], x.shape[1]), dtype=int)
    name_to_idx = {name: idx for idx, name in enumerate(columns)}
    for parent, child in dag.edges():
        adj[name_to_idx[parent], name_to_idx[child]] = 1
    return topological_sort_from_adj(adj)


def run_method(data: np.ndarray, method: str, args: argparse.Namespace) -> np.ndarray:
    if method == "direct_lingam":
        return run_direct_lingam(data, args)
    if method == "ica_lingam":
        return run_ica_lingam(data, args)
    if method == "hillclimb_bic":
        return run_hillclimb_bic(data, args)
    raise ValueError(f"Unknown method: {method}")


def evaluate_order(true_dag: np.ndarray, pred_order: np.ndarray) -> Dict[str, Any]:
    true_dag = remove_diag(true_dag).astype(int)
    pred_order = np.asarray(pred_order, dtype=int)
    valid, num_violations, violation_rate = is_valid_topological_order(true_dag, pred_order)
    num_edges = int(true_dag.sum())
    return {
        "topo_valid": float(valid),
        "num_violations": float(num_violations),
        "violation_rate": float(violation_rate),
        "edge_precedence_accuracy": edge_precedence_accuracy(true_dag, pred_order),
        "num_true_edges": float(num_edges),
    }


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    methods = parse_csv_list(args.methods)
    valid_methods = {"direct_lingam", "ica_lingam", "hillclimb_bic"}
    unknown = set(methods) - valid_methods
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}. Use any of: {sorted(valid_methods)}")

    sample_sizes = parse_int_list(args.sample_sizes)
    group_cols = ["benchmark_kind", "generator", "method", "sample_size"]
    if args.benchmark_kind == "fixed_graph":
        group_cols.extend(["graph_id", "graph_name"])
    summary = RunningSummary(group_cols)
    detailed_rows: List[Dict[str, Any]] = []

    print("=" * 100)
    print("Traditional causal-order evaluation")
    print(f"benchmark_kind: {args.benchmark_kind}")
    print(f"benchmark_root: {Path(args.benchmark_root).expanduser().resolve()}")
    print(f"methods:        {methods}")
    print(f"sample_sizes:   {sample_sizes}")
    print("=" * 100)

    for n in sample_sizes:
        h5_files = discover_h5_files(args, n)
        if args.max_files_per_n is not None:
            h5_files = h5_files[: args.max_files_per_n]
        if not h5_files:
            raise FileNotFoundError(f"No h5 files found for n={n}.")
        print("-" * 100)
        print(f"n={n}: files={len(h5_files)}")

        datasets_seen_for_n = 0
        for file_idx, (h5_path, meta) in enumerate(h5_files, start=1):
            with h5py.File(h5_path, "r") as f:
                data_arr = f["data"]
                label_arr = f["label"]
                num_datasets = data_arr.shape[0]
                if args.max_datasets_per_file is not None:
                    num_datasets = min(num_datasets, args.max_datasets_per_file)

                print(
                    f"  [{file_idx}/{len(h5_files)}] "
                    f"graph={meta.get('graph_id', -1)} generator={meta.get('generator')} "
                    f"{h5_path.name} datasets={num_datasets}/{data_arr.shape[0]}"
                )

                for data_idx in range(num_datasets):
                    if args.max_datasets_per_n is not None and datasets_seen_for_n >= args.max_datasets_per_n:
                        break
                    data = np.asarray(data_arr[data_idx], dtype=np.float64)
                    true_dag = remove_diag(np.asarray(label_arr[data_idx], dtype=int))
                    datasets_seen_for_n += 1

                    for method in methods:
                        group = {
                            "benchmark_kind": args.benchmark_kind,
                            "generator": meta.get("generator", args.distribution),
                            "method": method,
                            "sample_size": n,
                        }
                        if args.benchmark_kind == "fixed_graph":
                            group["graph_id"] = int(meta.get("graph_id", -1))
                            group["graph_name"] = str(meta.get("graph_name", "unknown"))

                        row: Dict[str, Any] = {
                            **group,
                            "h5_path": str(h5_path),
                            "dataset_index": data_idx,
                            "num_samples": int(data.shape[0]),
                            "num_nodes": int(data.shape[1]),
                        }
                        try:
                            start = time.time()
                            pred_order = run_method(data, method, args)
                            metrics = evaluate_order(true_dag, pred_order)
                            metrics["runtime_sec"] = float(time.time() - start)
                            row.update(metrics)
                            row["pred_order"] = order_to_str(pred_order)
                            summary.add_ok(group, metrics)
                        except Exception as exc:
                            row["error"] = repr(exc)
                            summary.add_error(group)
                            if args.print_errors:
                                print(f"    [ERROR] {method} n={n} idx={data_idx}: {repr(exc)}")
                        detailed_rows.append(row)

                if args.max_datasets_per_n is not None and datasets_seen_for_n >= args.max_datasets_per_n:
                    break

    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{args.summary_name}_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    summary_df = summary.to_frame()
    detailed_df = pd.DataFrame(detailed_rows)
    summary_path = results_dir / f"{args.summary_name}_summary.csv"
    detailed_path = results_dir / f"{args.summary_name}_detailed.csv"
    summary_df.to_csv(summary_path, index=False)
    detailed_df.to_csv(detailed_path, index=False)
    print("=" * 100)
    print(f"Wrote summary:  {summary_path}")
    print(f"Wrote detailed: {detailed_path}")
    print("=" * 100)
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate traditional causal-order baselines on 4-node benchmarks.")
    parser.add_argument("--methods", type=str, default="direct_lingam,ica_lingam")
    parser.add_argument("--benchmark_kind", type=str, default="random_gp", choices=["random_gp", "fixed_graph"])
    parser.add_argument("--benchmark_root", type=str, default="benchmark_data_4var")
    parser.add_argument("--distribution", type=str, default="csnp_gp_4var_ERL0U1")
    parser.add_argument("--fixed_generators", type=str, default="csnp_gp,rff_gaussian")
    parser.add_argument("--graph_ids", type=str, default="all")
    parser.add_argument("--num_nodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_sizes", type=str, default="5,20,50,100,300,1000,3000")
    parser.add_argument("--standardize", action="store_true", default=True)
    parser.add_argument("--no_standardize", action="store_false", dest="standardize")
    parser.add_argument("--direct_lingam_measure", type=str, default="pwling")
    parser.add_argument("--ica_lingam_max_iter", type=int, default=1000)
    parser.add_argument("--max_files_per_n", type=int, default=None)
    parser.add_argument("--max_datasets_per_file", type=int, default=None)
    parser.add_argument("--max_datasets_per_n", type=int, default=None)
    parser.add_argument("--print_errors", action="store_true")
    parser.add_argument("--results_dir", type=str, default="result/traditional_order_4var")
    parser.add_argument("--summary_name", type=str, default="traditional_order_4var")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
