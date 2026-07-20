#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a joint topo+skeleton model as CPDAG and compare with baselines.

The joint model is converted to a DAG by:
  1. predicting an unordered skeleton with the skeleton head,
  2. predicting one or more root-to-leaf topological orders with the topo head,
  3. orienting every predicted skeleton edge from earlier to later in the order.

The resulting DAG is converted to CPDAG and compared with the true CPDAG. Only
the two requested metrics are reported:
  - cpdag_exact_match
  - cpdag_shd_pairwise
"""

from __future__ import annotations

import argparse
import json
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

from evaluate_csnp_4var_gp_benchmark_summary import (  # noqa: E402
    compute_cpdag_metrics,
    cpdag_key,
    dag_to_cpdag_by_mec,
    discover_h5_files,
    most_common_item,
    parse_csv_list,
    parse_int_list,
    remove_diag,
    run_ges_cpdag,
    run_pc_cpdag,
)
from evaluate_topo_order_4var_summary import (  # noqa: E402
    encode_input,
    load_model as load_topo_model,
    sample_topo_diffusion_orders_from_repr,
)


KEEP_METRICS = ["cpdag_exact_match", "cpdag_shd_pairwise"]


class RunningSummary:
    def __init__(self, group_cols: Sequence[str]) -> None:
        self.group_cols = list(group_cols)
        self.sums: Dict[Tuple[Any, ...], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.counts: Dict[Tuple[Any, ...], int] = defaultdict(int)
        self.errors: Dict[Tuple[Any, ...], int] = defaultdict(int)

    def add_ok(self, group: Dict[str, Any], metrics: Dict[str, float]) -> None:
        key = tuple(group[c] for c in self.group_cols)
        self.counts[key] += 1
        for metric in KEEP_METRICS:
            value = metrics.get(metric)
            if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
                self.sums[key][metric] += float(value)

    def add_error(self, group: Dict[str, Any]) -> None:
        key = tuple(group[c] for c in self.group_cols)
        self.errors[key] += 1

    def to_frame(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        keys = sorted(set(self.counts) | set(self.errors))
        for key in keys:
            n_ok = self.counts.get(key, 0)
            row = {col: value for col, value in zip(self.group_cols, key)}
            row["n_ok"] = n_ok
            row["n_errors"] = self.errors.get(key, 0)
            for metric in KEEP_METRICS:
                total = self.sums.get(key, {}).get(metric, np.nan)
                row[f"mean_{metric}"] = total / n_ok if n_ok > 0 else np.nan
            rows.append(row)
        return pd.DataFrame(rows)


def skeleton_prob_from_node_repr(
    model: Any,
    node_repr: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
) -> np.ndarray:
    if not hasattr(model, "_skeleton_logits_from_node_repr"):
        raise ValueError("Joint DAG evaluation requires a model with _skeleton_logits_from_node_repr.")

    with torch.no_grad():
        logits = model._skeleton_logits_from_node_repr(node_repr, mask=mask)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy().astype(np.float64)

    probs = np.maximum(probs, probs.T)
    np.fill_diagonal(probs, 0.0)
    return probs


def orient_skeleton_by_order(skeleton_prob: np.ndarray, order: np.ndarray, threshold: float) -> np.ndarray:
    order = np.asarray(order, dtype=int)
    num_nodes = skeleton_prob.shape[0]
    if sorted(order.tolist()) != list(range(num_nodes)):
        raise ValueError(f"Invalid order: {order.tolist()}")

    position = np.empty(num_nodes, dtype=int)
    position[order] = np.arange(num_nodes)

    dag = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if max(float(skeleton_prob[i, j]), float(skeleton_prob[j, i])) > threshold:
                if position[i] < position[j]:
                    dag[i, j] = 1
                else:
                    dag[j, i] = 1
    return dag


def orient_skeleton_by_layer_id(
    skeleton_prob: np.ndarray,
    layer_id: np.ndarray,
    threshold: float,
) -> np.ndarray:
    layer_id = np.asarray(layer_id, dtype=int)
    num_nodes = skeleton_prob.shape[0]
    dag = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if max(float(skeleton_prob[i, j]), float(skeleton_prob[j, i])) > threshold:
                if layer_id[i] < layer_id[j]:
                    dag[i, j] = 1
                elif layer_id[j] < layer_id[i]:
                    dag[j, i] = 1
    return dag


def joint_predict_cpdag(
    model: Any,
    data: np.ndarray,
    *,
    device: str,
    standardize: bool,
    threshold: float,
    order_mode: str,
    num_order_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    inputs = encode_input(data, device=device, standardize=standardize)
    with torch.no_grad():
        node_repr = model._encode_raw_data(inputs, mask=None)
    skeleton_prob = skeleton_prob_from_node_repr(model, node_repr, mask=None)
    if hasattr(model, "_decode_source_layers_from_node_repr") and not hasattr(model, "reverse_model"):
        valid_nodes = torch.ones(
            node_repr.shape[:2],
            dtype=torch.bool,
            device=node_repr.device,
        )
        with torch.no_grad():
            layer_ids, _, _ = model._decode_source_layers_from_node_repr(
                node_repr,
                valid_nodes=valid_nodes,
            )
        pred_dag = orient_skeleton_by_layer_id(
            skeleton_prob,
            layer_ids[0].detach().cpu().numpy(),
            threshold,
        )
        pred_dir, pred_undir, _ = dag_to_cpdag_by_mec(pred_dag)
        return pred_dir, pred_undir

    orders = sample_topo_diffusion_orders_from_repr(
        model=model,
        node_repr=node_repr,
        num_order_samples=num_order_samples,
        deterministic=(order_mode == "deterministic"),
        beam=(order_mode == "beam"),
    )
    orders = np.asarray(orders, dtype=int).reshape(-1, skeleton_prob.shape[0])

    cpdag_counts: Dict[Any, int] = defaultdict(int)
    cpdag_by_key: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
    for order in orders:
        pred_dag = orient_skeleton_by_order(skeleton_prob, order, threshold)
        pred_dir, pred_undir, _ = dag_to_cpdag_by_mec(pred_dag)
        key = cpdag_key(pred_dir, pred_undir)
        cpdag_counts[key] += 1
        cpdag_by_key[key] = (pred_dir, pred_undir)

    return cpdag_by_key[most_common_item(cpdag_counts)]


def add_cpdag_result(
    summary: RunningSummary,
    group: Dict[str, Any],
    true_dir: np.ndarray,
    true_undir: np.ndarray,
    pred_dir: np.ndarray,
    pred_undir: np.ndarray,
) -> None:
    metrics = compute_cpdag_metrics(true_dir, true_undir, pred_dir, pred_undir)
    summary.add_ok(group, metrics)


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    methods = set(parse_csv_list(args.methods))
    valid_methods = {"joint", "pc_fisherz", "pc_kci", "ges"}
    unknown = methods - valid_methods
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}. Use any of {sorted(valid_methods)}.")

    sample_sizes = parse_int_list(args.sample_sizes)
    models_root = Path(args.models_root).expanduser().resolve()
    bak_path = Path(args.bak_model_file).expanduser().resolve()

    joint_model = None
    joint_variant = ""
    if "joint" in methods:
        print(f"Loading joint model: {args.joint_run_name}/{args.joint_checkpoint}")
        joint_model, config, loaded_path = load_topo_model(
            models_root=models_root,
            run_name=args.joint_run_name,
            checkpoint=args.joint_checkpoint,
            source="current",
            device=device,
            bak_path=bak_path,
        )
        has_order_sampler = (
            hasattr(joint_model, "reverse_model")
            or hasattr(joint_model, "_decode_source_layers_from_node_repr")
        )
        if not hasattr(joint_model, "_skeleton_logits_from_node_repr") or not has_order_sampler:
            raise ValueError(
                "The joint method requires a topo/source-layer model with a skeleton head "
                f"(loaded module={config.get('module')!r})."
            )
        joint_variant = (
            f"{args.joint_run_name}_{args.joint_checkpoint}_"
            f"{args.joint_order_mode}_orders{args.num_order_samples}_thr{args.threshold}"
        )
        print(f"  loaded: {loaded_path}")

    group_cols = ["benchmark", "generator", "graph_id", "graph_name", "num_samples", "method", "variant"]
    summary = RunningSummary(group_cols)

    print("=" * 100)
    print("Joint DAG -> CPDAG vs traditional CPDAG evaluation")
    print(f"benchmark_root: {Path(args.benchmark_root).expanduser().resolve()}")
    print(f"models_root:    {models_root}")
    print(f"benchmark_kind: {args.benchmark_kind}")
    print(f"methods:        {sorted(methods)}")
    print(f"sample_sizes:   {sample_sizes}")
    print(f"device:         {device}")
    print("=" * 100)

    start = time.time()
    for n in sample_sizes:
        files = discover_h5_files(args, n)
        if args.max_files_per_n is not None:
            files = files[: args.max_files_per_n]
        if not files:
            if args.skip_missing:
                print(f"[SKIP] no h5 files for n={n}")
                continue
            raise FileNotFoundError(f"No h5 files found for n={n}.")

        print("-" * 100)
        print(f"n={n}: files={len(files)}")
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
                    true_dir, true_undir, _ = dag_to_cpdag_by_mec(true_dag)
                    seen_for_n += 1

                    base_group = {
                        "benchmark": meta["benchmark"],
                        "generator": meta["generator"],
                        "graph_id": meta["graph_id"],
                        "graph_name": meta["graph_name"],
                        "num_samples": n,
                    }

                    if "joint" in methods and joint_model is not None:
                        group = {
                            **base_group,
                            "method": args.joint_method_name,
                            "variant": joint_variant,
                        }
                        try:
                            pred_dir, pred_undir = joint_predict_cpdag(
                                joint_model,
                                data,
                                device=device,
                                standardize=args.standardize_joint,
                                threshold=args.threshold,
                                order_mode=args.joint_order_mode,
                                num_order_samples=args.num_order_samples,
                            )
                            add_cpdag_result(summary, group, true_dir, true_undir, pred_dir, pred_undir)
                        except Exception as exc:
                            summary.add_error(group)
                            if args.print_errors:
                                print(f"    [ERROR] joint n={n} idx={data_idx}: {repr(exc)}")

                    if "pc_fisherz" in methods:
                        group = {**base_group, "method": "PC-fisherz", "variant": f"alpha{args.pc_alpha}"}
                        old_test = args.pc_test
                        args.pc_test = "fisherz"
                        try:
                            pred_dir, pred_undir = run_pc_cpdag(data, args)
                            add_cpdag_result(summary, group, true_dir, true_undir, pred_dir, pred_undir)
                        except Exception as exc:
                            summary.add_error(group)
                            if args.print_errors:
                                print(f"    [ERROR] PC-fisherz n={n} idx={data_idx}: {repr(exc)}")
                        finally:
                            args.pc_test = old_test

                    if "pc_kci" in methods:
                        group = {**base_group, "method": "PC-kci", "variant": f"alpha{args.pc_alpha}"}
                        old_test = args.pc_test
                        args.pc_test = "kci"
                        try:
                            pred_dir, pred_undir = run_pc_cpdag(data, args)
                            add_cpdag_result(summary, group, true_dir, true_undir, pred_dir, pred_undir)
                        except Exception as exc:
                            summary.add_error(group)
                            if args.print_errors:
                                print(f"    [ERROR] PC-kci n={n} idx={data_idx}: {repr(exc)}")
                        finally:
                            args.pc_test = old_test

                    if "ges" in methods:
                        group = {**base_group, "method": "GES-BIC", "variant": args.ges_score_func}
                        try:
                            pred_dir, pred_undir = run_ges_cpdag(data, args)
                            add_cpdag_result(summary, group, true_dir, true_undir, pred_dir, pred_undir)
                        except Exception as exc:
                            summary.add_error(group)
                            if args.print_errors:
                                print(f"    [ERROR] GES n={n} idx={data_idx}: {repr(exc)}")

                    if args.progress_every > 0 and seen_for_n % args.progress_every == 0:
                        print(f"    done n={n}: {seen_for_n}")

                if args.max_datasets_per_n is not None and seen_for_n >= args.max_datasets_per_n:
                    break

    print("=" * 100)
    print(f"Finished. elapsed={time.time() - start:.1f}s")
    return summary.to_frame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate joint DAG CPDAG metrics against PC/GES baselines.")

    parser.add_argument("--benchmark_root", type=str, default="benchmark_data_4var")
    parser.add_argument("--benchmark_kind", type=str, default="random_gp", choices=["random_gp", "fixed_graph"])
    parser.add_argument("--distribution", type=str, default="csnp_gp_4var_ERL0U1")
    parser.add_argument("--fixed_generators", type=str, default="csnp_gp,rff_gaussian")
    parser.add_argument("--graph_ids", type=str, default="all")
    parser.add_argument("--num_nodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_sizes", type=str, default="5,20,50,100,300,1000,3000")
    parser.add_argument("--methods", type=str, default="joint,pc_fisherz,pc_kci,ges")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--models_root", type=str, default="ml2_meta_causal_discovery/experiments/causal_classification/models")
    parser.add_argument("--joint_run_name", type=str, required=True)
    parser.add_argument("--joint_checkpoint", type=str, required=True)
    parser.add_argument("--joint_method_name", type=str, default="JointTopoSkeleton")
    parser.add_argument("--joint_order_mode", type=str, default="beam", choices=["sample", "deterministic", "beam"])
    parser.add_argument("--num_order_samples", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--standardize_joint", action="store_true", default=True)
    parser.add_argument("--no_standardize_joint", action="store_false", dest="standardize_joint")
    parser.add_argument(
        "--bak_model_file",
        type=str,
        default="ml2_meta_causal_discovery/models/causaltransformernp.py.mask_version.bak",
    )

    parser.add_argument("--standardize_traditional", action="store_true", default=True)
    parser.add_argument("--no_standardize_traditional", action="store_false", dest="standardize_traditional")
    parser.add_argument("--pc_alpha", type=float, default=0.05)
    parser.add_argument("--pc_test", type=str, default="fisherz", choices=["fisherz", "kci", "chisq", "gsq"])
    parser.add_argument("--pc_stable", action="store_true", default=True)
    parser.add_argument("--pc_uc_rule", type=int, default=0)
    parser.add_argument("--pc_uc_priority", type=int, default=2)
    parser.add_argument("--ges_score_func", type=str, default="local_score_BIC")

    parser.add_argument("--results_dir", type=str, default="result/joint_dag_cpdag_vs_traditional")
    parser.add_argument("--summary_name", type=str, default="summary_cpdag_joint_vs_traditional.csv")
    parser.add_argument("--max_files_per_n", type=int, default=None)
    parser.add_argument("--max_datasets_per_file", type=int, default=None)
    parser.add_argument("--max_datasets_per_n", type=int, default=None)
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--progress_every", type=int, default=200)
    parser.add_argument("--print_errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    summary = evaluate(args)
    keep_cols = [
        "benchmark",
        "generator",
        "graph_id",
        "graph_name",
        "num_samples",
        "method",
        "variant",
        "n_ok",
        "n_errors",
        "mean_cpdag_exact_match",
        "mean_cpdag_shd_pairwise",
    ]
    summary = summary[[col for col in keep_cols if col in summary.columns]]
    out_path = results_dir / args.summary_name
    summary.to_csv(out_path, index=False)
    print(f"Wrote CPDAG summary: {out_path}")


if __name__ == "__main__":
    main()
