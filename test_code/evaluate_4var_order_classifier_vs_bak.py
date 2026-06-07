#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the 4-node direct order classifier against bak Q samples.

The classifier enumerates all 4! orders, so besides sampling orders for a fair
comparison with bak, this script also reports the exact probability mass that
the classifier assigns to valid topological orders.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from test_code.train_4var_order_classifier import (  # noqa: E402
    FourNodeOrderClassifier,
    all_permutation_orders,
    edge_precedence_accuracy,
    valid_order_mask,
)


def load_classifier(
    *,
    run_dir: Path,
    checkpoint: str,
    device: str,
) -> Tuple[FourNodeOrderClassifier, Dict[str, Any], Path]:
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / checkpoint
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = FourNodeOrderClassifier(
        d_model=config["dim_model"],
        dim_feedforward=config["dim_feedforward"],
        nhead=config["nhead"],
        num_layers_encoder=config["num_layers_encoder"],
        num_nodes=config["num_nodes"],
        dropout=config.get("dropout", 0.0),
        scorer_hidden=config.get("scorer_hidden", None),
        use_positional_encoding=config.get("use_positional_encoding", False),
        device=device,
        dtype=torch.float32,
    ).to(device)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, checkpoint_path


def classifier_orders_and_metrics(
    *,
    model: FourNodeOrderClassifier,
    data: np.ndarray,
    true_dag: np.ndarray,
    orders_24: torch.Tensor,
    num_order_samples: int,
    device: str,
    standardize: bool,
) -> Tuple[np.ndarray, Dict[str, float]]:
    inputs = encode_input(data, device=device, standardize=standardize)
    target = torch.tensor(true_dag, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        scores = model(inputs, orders=orders_24, mask=None)
        probs = torch.softmax(scores.float(), dim=-1)
        valid = valid_order_mask(target, orders_24)
        top_idx = scores.argmax(dim=-1)
        top_orders = orders_24[top_idx]
        sample_idx = torch.multinomial(probs, num_samples=num_order_samples, replacement=True)
        sampled_orders = orders_24[sample_idx[0]]

        valid_mass = (probs * valid.float()).sum(dim=-1)[0]
        top1_valid = valid[0, top_idx[0]].float()
        top1_edge_acc = edge_precedence_accuracy(target, top_orders)[0]
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)[0]
        top1_prob = probs[0, top_idx[0]]

    exact_metrics = {
        "valid_mass": float(valid_mass.item()),
        "invalid_mass": float((1.0 - valid_mass).item()),
        "order_nll": float((-valid_mass.clamp_min(1e-12).log()).item()),
        "top1_valid": float(top1_valid.item()),
        "top1_edge_precedence_accuracy": float(top1_edge_acc.item()),
        "top1_prob": float(top1_prob.item()),
        "entropy": float(entropy.item()),
        "num_valid_orders_exact": float(valid.float().sum(dim=-1)[0].item()),
    }
    return sampled_orders.detach().cpu().numpy().astype(int), exact_metrics


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    requested_methods = set(parse_csv_list(args.methods))
    valid_methods = {"classifier", "bak"}
    unknown = requested_methods - valid_methods
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}. Use any of: {sorted(valid_methods)}")

    classifier_run_dir = Path(args.classifier_results_dir).expanduser().resolve() / args.classifier_run_name
    models_root = Path(args.models_root).expanduser().resolve()
    bak_path = Path(args.bak_model_file).expanduser().resolve()
    sample_sizes = parse_int_list(args.sample_sizes)

    loaded: List[Tuple[str, str, str, Any]] = []
    orders_24 = all_permutation_orders(args.num_nodes, torch.device(device))

    print("=" * 100)
    print("4-node order-classifier vs bak topology evaluation")
    print(f"benchmark_kind:     {args.benchmark_kind}")
    print(f"benchmark_root:     {Path(args.benchmark_root).expanduser().resolve()}")
    print(f"sample_sizes:       {sample_sizes}")
    print(f"num_order_samples:  {args.num_order_samples}")
    print(f"device:             {device}")
    print("=" * 100)

    if "classifier" in requested_methods:
        classifier, _, classifier_path = load_classifier(
            run_dir=classifier_run_dir,
            checkpoint=args.classifier_checkpoint,
            device=device,
        )
        loaded.append((
            args.classifier_method_name,
            "classifier",
            f"{args.classifier_run_name}_{args.classifier_checkpoint}_orders{args.num_order_samples}",
            classifier,
        ))
        print(f"Loaded classifier: {classifier_path}")

    if "bak" in requested_methods:
        bak_model, _, bak_model_path = load_model(
            models_root=models_root,
            run_name=args.bak_run_name,
            checkpoint=args.bak_checkpoint,
            source="bak",
            device=device,
            bak_path=bak_path,
        )
        loaded.append((
            args.bak_method_name,
            "bak",
            f"{args.bak_run_name}_{args.bak_checkpoint}_orders{args.num_order_samples}",
            bak_model,
        ))
        print(f"Loaded bak: {bak_model_path}")

    group_cols = ["benchmark", "generator", "graph_id", "graph_name", "num_samples", "method", "variant"]
    summary = RunningSummary(group_cols)
    start = time.time()
    for n in sample_sizes:
        files = discover_h5_files(args, n)
        if args.max_files_per_n is not None:
            files = files[: args.max_files_per_n]
        if not files:
            if args.skip_missing:
                print(f"[SKIP] no files for n={n}")
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
                    seen_for_n += 1

                    for method, kind, variant, model in loaded:
                        group = {
                            "benchmark": meta["benchmark"],
                            "generator": meta["generator"],
                            "graph_id": meta["graph_id"],
                            "graph_name": meta["graph_name"],
                            "num_samples": n,
                            "method": method,
                            "variant": variant,
                        }
                        try:
                            if kind == "classifier":
                                sampled_orders, exact_metrics = classifier_orders_and_metrics(
                                    model=model,
                                    data=data,
                                    true_dag=true_dag,
                                    orders_24=orders_24,
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
                            summary.add_ok(group, metrics)
                        except Exception as e:
                            summary.add_error(group)
                            if args.print_errors:
                                print(f"    [ERROR] {method} n={n} idx={data_idx}: {repr(e)}")
                if args.max_datasets_per_n is not None and seen_for_n >= args.max_datasets_per_n:
                    break

    print("=" * 100)
    print(f"Finished. elapsed={time.time() - start:.1f}s")
    return summary.to_frame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 4-node direct order classifier against bak.")
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
    parser.add_argument("--classifier_run_name", type=str, default="order_classifier_4var_d128_bs16")
    parser.add_argument("--classifier_checkpoint", type=str, default="model_13.pt")
    parser.add_argument("--classifier_method_name", type=str, default="OrderClassifier-bs16")

    parser.add_argument("--models_root", type=str, default="ml2_meta_causal_discovery/experiments/causal_classification/models")
    parser.add_argument("--bak_run_name", type=str, default="gp_4var_prob_bakL_100k")
    parser.add_argument("--bak_checkpoint", type=str, default="model_9.pt")
    parser.add_argument("--bak_method_name", type=str, default="CSNP-bakL-model9")
    parser.add_argument("--bak_model_file", type=str, default="ml2_meta_causal_discovery/models/causaltransformernp.py.mask_version.bak")

    parser.add_argument("--results_dir", type=str, default="result/order_classifier_4var_eval")
    parser.add_argument("--summary_name", type=str, default="summary_order_classifier_vs_bak.csv")
    parser.add_argument("--max_files_per_n", type=int, default=None)
    parser.add_argument("--max_datasets_per_file", type=int, default=None)
    parser.add_argument("--max_datasets_per_n", type=int, default=None)
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--print_errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluate(args)
    out_path = results_dir / args.summary_name
    summary.to_csv(out_path, index=False)
    print(f"Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
