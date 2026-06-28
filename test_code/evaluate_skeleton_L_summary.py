#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate whether CSNP L parameters learn the undirected skeleton.

This script intentionally ignores sampled permutations Q. It decodes L_param,
converts sigmoid(L_param) into undirected pair probabilities, thresholds those
probabilities, and compares them with the true graph skeleton.

The summary keeps only the key skeleton metrics:
  - exact match, SHD, F1, AUC, and Brier score.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import pandas as pd
import torch as th


KEY_METRICS = [
    "skeleton_exact_match",
    "skeleton_shd",
    "skeleton_f1",
    "skeleton_auc",
    "skeleton_brier",
]
PLOT_METRICS = [f"mean_{metric}" for metric in KEY_METRICS]


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in parse_csv_list(s)]


def parse_checkpoints(
    *,
    explicit: str,
    single: str,
    start: Optional[int],
    end: Optional[int],
    prefix: str,
    suffix: str,
) -> List[str]:
    if explicit:
        checkpoints = [x.strip() for x in explicit.split(",") if x.strip()]
    elif start is not None or end is not None:
        if start is None or end is None:
            raise ValueError("Use both checkpoint_start and checkpoint_end for checkpoint sweeps.")
        checkpoints = [f"{prefix}{i}{suffix}" for i in range(start, end + 1)]
    else:
        checkpoints = [single]
    if not checkpoints:
        raise ValueError("No checkpoints selected.")
    return checkpoints


def checkpoint_index(checkpoint: str) -> int:
    match = re.search(r"(\d+)(?=\.pt$|$)", checkpoint)
    return int(match.group(1)) if match else -1


def pair_order(num_nodes: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]


def remove_diag(g: np.ndarray) -> np.ndarray:
    out = np.asarray(g).copy()
    np.fill_diagonal(out, 0)
    return out


def skeleton_from_dag(dag: np.ndarray) -> np.ndarray:
    dag = remove_diag(dag).astype(int)
    skel = ((dag + dag.T) > 0).astype(int)
    np.fill_diagonal(skel, 0)
    return skel


def standardize_batch(x: np.ndarray, standardize: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if not standardize:
        return x
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)


def binary_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn > 0 else np.nan
    return {
        "skeleton_tp": float(tp),
        "skeleton_fp": float(fp),
        "skeleton_fn": float(fn),
        "skeleton_tn": float(tn),
        "skeleton_precision": float(precision),
        "skeleton_recall": float(recall),
        "skeleton_f1": float(f1),
        "skeleton_accuracy": float(accuracy),
    }


def roc_auc_score_binary(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    score = np.asarray(score, dtype=float).reshape(-1)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and score[order[j]] == score[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos_ranks = ranks[y_true == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision_binary(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    score = np.asarray(score, dtype=float).reshape(-1)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score)
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted == 1)
    precision_at_k = tp_cum / (np.arange(len(y_sorted)) + 1)
    return float(precision_at_k[y_sorted == 1].sum() / n_pos)


def skeleton_metrics(true_dag: np.ndarray, skel_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    true_skel = skeleton_from_dag(true_dag)
    d = true_skel.shape[0]
    pairs = pair_order(d)
    y_true = np.array([true_skel[i, j] for i, j in pairs], dtype=int)
    y_prob = np.array([skel_prob[i, j] for i, j in pairs], dtype=float)
    y_pred = (y_prob > threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = 2 * tp + fp + fn
    skeleton_f1 = 1.0 if denom == 0 else (2 * tp / denom)

    return {
        "skeleton_exact_match": float(np.array_equal(y_true, y_pred)),
        "skeleton_shd": float(np.abs(y_true - y_pred).sum()),
        "skeleton_auc": roc_auc_score_binary(y_true, y_prob),
        "skeleton_f1": float(skeleton_f1),
        "skeleton_brier": float(np.mean((y_prob - y_true) ** 2)),
    }


class RunningSummary:
    def __init__(self, group_cols: Sequence[str]) -> None:
        self.group_cols = list(group_cols)
        self.sums: Dict[Tuple[Any, ...], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.metric_counts: Dict[Tuple[Any, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.counts: Dict[Tuple[Any, ...], int] = defaultdict(int)
        self.errors: Dict[Tuple[Any, ...], int] = defaultdict(int)

    def add_ok(self, group: Dict[str, Any], metrics: Dict[str, float]) -> None:
        key = tuple(group[c] for c in self.group_cols)
        self.counts[key] += 1
        for metric, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
                self.sums[key][metric] += float(value)
                self.metric_counts[key][metric] += 1

    def add_error(self, group: Dict[str, Any]) -> None:
        key = tuple(group[c] for c in self.group_cols)
        self.errors[key] += 1

    def to_frame(self) -> pd.DataFrame:
        keys = sorted(set(self.counts) | set(self.errors))
        rows: List[Dict[str, Any]] = []
        for key in keys:
            n_ok = self.counts.get(key, 0)
            row = {col: value for col, value in zip(self.group_cols, key)}
            row["n_ok"] = n_ok
            row["n_errors"] = self.errors.get(key, 0)
            for metric in KEY_METRICS:
                total = self.sums.get(key, {}).get(metric, 0.0)
                metric_count = self.metric_counts.get(key, {}).get(metric, 0)
                row[f"mean_{metric}"] = total / metric_count if metric_count > 0 else np.nan
            rows.append(row)
        return pd.DataFrame(rows)


def read_h5_meta(h5_path: Path, fallback: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(fallback)
    with h5py.File(h5_path, "r") as f:
        for key, value in f.attrs.items():
            if isinstance(value, np.generic):
                value = value.item()
            meta[key] = value
    if "generator" not in meta and "mechanism" in meta:
        meta["generator"] = meta["mechanism"]
    return meta


def graph_id_from_dir(graph_dir: Path) -> int:
    match = re.match(r"graph_(\d+)__", graph_dir.name)
    if match is None:
        raise ValueError(f"Could not parse graph id from {graph_dir}")
    return int(match.group(1))


def discover_h5_files(args: argparse.Namespace, sample_size: int) -> List[Tuple[Path, Dict[str, Any]]]:
    root = Path(args.benchmark_root).expanduser().resolve()
    if args.benchmark_kind == "random_gp":
        pattern = root / args.distribution / f"n_{sample_size}" / f"seed_{args.seed}" / "*.h5"
        fallback = {
            "benchmark_kind": "random_gp",
            "generator": args.distribution,
            "graph_id": -1,
            "graph_name": "random",
            "sample_size": sample_size,
        }
        return [(Path(p), read_h5_meta(Path(p), fallback)) for p in sorted(glob.glob(str(pattern)))]

    if args.benchmark_kind == "fixed_graph":
        graph_dirs = sorted((root / "4").glob("graph_*__*"))
        if args.graph_ids.lower() != "all":
            keep = set(parse_int_list(args.graph_ids))
            graph_dirs = [g for g in graph_dirs if graph_id_from_dir(g) in keep]

        out: List[Tuple[Path, Dict[str, Any]]] = []
        for graph_dir in graph_dirs:
            gid = graph_id_from_dir(graph_dir)
            for generator in parse_csv_list(args.fixed_generators):
                candidate_generators = [generator]
                if generator == "rff":
                    candidate_generators.append("rff_gaussian")
                for folder in candidate_generators:
                    pattern = graph_dir / folder / f"n_{sample_size}" / f"seed_{args.seed}" / "*.h5"
                    matches = sorted(glob.glob(str(pattern)))
                    if matches:
                        break
                else:
                    continue

                fallback = {
                    "benchmark_kind": "fixed_graph",
                    "generator": folder,
                    "graph_id": gid,
                    "graph_name": graph_dir.name,
                    "sample_size": sample_size,
                }
                out.extend((Path(p), read_h5_meta(Path(p), fallback)) for p in matches)
        return out

    raise ValueError(f"Unsupported benchmark_kind: {args.benchmark_kind}")


def import_decoder_classes(source: str, bak_path: Path) -> Tuple[Any, Optional[Any]]:
    if source == "current":
        from ml2_meta_causal_discovery.models.causaltransformernp import (
            CausalProbabilisticARDecoder,
            CausalProbabilisticDecoder,
        )

        return CausalProbabilisticDecoder, CausalProbabilisticARDecoder

    if source == "bak":
        loader = SourceFileLoader("csnp_bak_causaltransformernp_for_skeleton_L", str(bak_path))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load bak model file: {bak_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.CausalProbabilisticDecoder, getattr(module, "CausalProbabilisticARDecoder", None)

    raise ValueError(f"Unknown model source: {source}")


def load_model(
    *,
    models_root: Path,
    run_name: str,
    checkpoint: str,
    source: str,
    bak_path: Path,
    device: str,
) -> Tuple[Any, Dict[str, Any], Path]:
    model_dir = models_root / run_name
    config_path = model_dir / "config.json"
    model_path = model_dir / checkpoint
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    module = config.get("module", "probabilistic")
    if module in {
        "topo_skeleton",
        "topo_diffusion_skeleton",
        "topo_priority_diffusion_skeleton",
        "topo_diffusion_skeleton_single_encoder",
        "topo_priority_diffusion_skeleton_single_encoder",
    }:
        from ml2_meta_causal_discovery.models.topo_order_diffusion import (
            CausalPriorityTopoOrderDiffusionSingleEncoderWithSkeleton,
            CausalSkeletonDecoder,
            CausalPriorityTopoOrderDiffusionWithSkeleton,
            CausalTopoOrderDiffusionSingleEncoderWithSkeleton,
            CausalTopoOrderDiffusionWithSkeleton,
        )

        if module == "topo_skeleton":
            topo_cls = CausalSkeletonDecoder
        elif module == "topo_priority_diffusion_skeleton":
            topo_cls = CausalPriorityTopoOrderDiffusionWithSkeleton
        elif module == "topo_diffusion_skeleton_single_encoder":
            topo_cls = CausalTopoOrderDiffusionSingleEncoderWithSkeleton
        elif module == "topo_priority_diffusion_skeleton_single_encoder":
            topo_cls = CausalPriorityTopoOrderDiffusionSingleEncoderWithSkeleton
        else:
            topo_cls = CausalTopoOrderDiffusionWithSkeleton
        model = topo_cls(
            d_model=config["d_model"],
            emb_depth=1,
            dim_feedforward=config["dim_feedforward"],
            nhead=config["nhead"],
            dropout=config.get("dropout", 0.0),
            num_layers_encoder=config["num_layers_encoder"],
            num_layers_decoder=config["num_layers_decoder"],
            num_nodes=config["num_nodes"],
            n_perm_samples=config.get("n_perm_samples", 25),
            sinkhorn_iter=config.get("sinkhorn_iter", 300),
            use_positional_encoding=config["use_positional_encoding"],
            topo_num_timesteps=config.get("topo_num_timesteps", 7),
            topo_sample_N=config.get("topo_sample_N", 1),
            topo_transition=config.get("topo_transition", "riffle"),
            topo_reverse=config.get("topo_reverse", "generalized_PL"),
            topo_reverse_steps=config.get("topo_reverse_steps", None),
            topo_beam_size=config.get("topo_beam_size", 20),
            topo_priority_scale_init=config.get("topo_priority_scale_init", -2.0),
            skeleton_loss_weight=config.get("skeleton_loss_weight", 1.0),
            order_loss_weight=config.get("order_loss_weight", 1.0),
            skeleton_decoder_layers=config.get("skeleton_decoder_layers", 2),
            device=device,
            dtype=th.float32,
        ).to(device)
    else:
        CausalProbabilisticDecoder, CausalProbabilisticARDecoder = import_decoder_classes(source, bak_path)
        kwargs = dict(
            d_model=config["d_model"],
            emb_depth=1,
            dim_feedforward=config["dim_feedforward"],
            nhead=config["nhead"],
            dropout=0.0,
            num_layers_encoder=config["num_layers_encoder"],
            num_layers_decoder=config["num_layers_decoder"],
            num_nodes=config["num_nodes"],
            n_perm_samples=config.get("n_perm_samples", 25),
            sinkhorn_iter=config.get("sinkhorn_iter", 300),
            use_positional_encoding=config["use_positional_encoding"],
            device=device,
            dtype=th.float32,
        )
        if module == "probabilistic":
            model = CausalProbabilisticDecoder(**kwargs).to(device)
        elif module == "probabilistic_ar":
            if CausalProbabilisticARDecoder is None:
                raise ValueError(f"{source} model source does not contain CausalProbabilisticARDecoder.")
            model = CausalProbabilisticARDecoder(
                **kwargs,
                num_topo_order_samples=config.get("num_topo_order_samples", 8),
                ar_hidden_dim=config.get("ar_hidden_dim", None),
            ).to(device)
        else:
            raise ValueError(
                "This script supports probabilistic/probabilistic_ar and "
                f"*_skeleton topo models, got module={module!r}."
            )

    try:
        state_dict = th.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = th.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, model_path


def decode_l_probs_batch(
    model: Any,
    data_batch: np.ndarray,
    *,
    standardize: bool,
    device: str,
) -> np.ndarray:
    x = standardize_batch(data_batch, standardize=standardize)
    inputs = th.tensor(x, dtype=th.float32, device=device)
    with th.no_grad():
        if hasattr(model, "_skeleton_logits_from_data"):
            logits = model._skeleton_logits_from_data(inputs, mask=None)
            edge_probs = th.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
        elif hasattr(model, "_encode_decode"):
            L_param, _, _ = model._encode_decode(inputs, mask=None)
            edge_probs = th.sigmoid(L_param).detach().cpu().numpy().astype(np.float64)
        else:
            if inputs.dim() == 3:
                target_data = inputs.unsqueeze(-1)
            else:
                target_data = inputs
            representation = model.encode(target_data=target_data, mask=None).squeeze(2)
            L_param, _ = model.decode(representation=representation, mask=None)
            edge_probs = th.sigmoid(L_param).detach().cpu().numpy().astype(np.float64)
    for b in range(edge_probs.shape[0]):
        np.fill_diagonal(edge_probs[b], 0.0)
    return edge_probs


def skeleton_prob_from_l(edge_probs: np.ndarray) -> np.ndarray:
    skel_prob = np.maximum(edge_probs, edge_probs.T)
    np.fill_diagonal(skel_prob, 0.0)
    return skel_prob


def method_specs(args: argparse.Namespace) -> List[Dict[str, str]]:
    specs: List[Dict[str, str]] = []
    methods = parse_csv_list(args.methods)
    if "bak" in methods:
        checkpoints = parse_checkpoints(
            explicit=args.bak_checkpoints,
            single=args.bak_checkpoint,
            start=args.bak_checkpoint_start,
            end=args.bak_checkpoint_end,
            prefix=args.bak_checkpoint_prefix,
            suffix=args.bak_checkpoint_suffix,
        )
        for checkpoint in checkpoints:
            specs.append(
                {
                    "method": "bak",
                    "run_name": args.bak_run_name,
                    "checkpoint": checkpoint,
                    "checkpoint_index": checkpoint_index(checkpoint),
                    "source": args.bak_model_source,
                }
            )
    if "ar" in methods:
        checkpoints = parse_checkpoints(
            explicit=args.ar_checkpoints,
            single=args.ar_checkpoint,
            start=args.ar_checkpoint_start,
            end=args.ar_checkpoint_end,
            prefix=args.ar_checkpoint_prefix,
            suffix=args.ar_checkpoint_suffix,
        )
        for checkpoint in checkpoints:
            specs.append(
                {
                    "method": "ar",
                    "run_name": args.ar_run_name,
                    "checkpoint": checkpoint,
                    "checkpoint_index": checkpoint_index(checkpoint),
                    "source": args.ar_model_source,
                }
            )
    if "topo_skeleton" in methods:
        checkpoints = parse_checkpoints(
            explicit=args.topo_checkpoints,
            single=args.topo_checkpoint,
            start=args.topo_checkpoint_start,
            end=args.topo_checkpoint_end,
            prefix=args.topo_checkpoint_prefix,
            suffix=args.topo_checkpoint_suffix,
        )
        for checkpoint in checkpoints:
            specs.append(
                {
                    "method": args.topo_method_name,
                    "run_name": args.topo_run_name,
                    "checkpoint": checkpoint,
                    "checkpoint_index": checkpoint_index(checkpoint),
                    "source": "current",
                }
            )
    unknown = sorted(set(methods) - {"bak", "ar", "topo_skeleton"})
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Use bak, ar, topo_skeleton, or a comma-separated subset.")
    return specs


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    work_dir = Path(args.work_dir).expanduser().resolve()
    models_root = (
        Path(args.models_root).expanduser().resolve()
        if args.models_root
        else work_dir / "experiments" / "causal_classification" / "models"
    )
    bak_path = Path(args.bak_path).expanduser().resolve()
    result_summary = RunningSummary(
        [
            "method",
            "run_name",
            "checkpoint",
            "checkpoint_index",
            "benchmark_kind",
            "generator",
            "graph_id",
            "graph_name",
            "sample_size",
        ]
    )

    loaded = []
    print(f"models_root:    {models_root}")
    for spec in method_specs(args):
        print(f"Loading {spec['method']}: {spec['run_name']}/{spec['checkpoint']} source={spec['source']}")
        try:
            model, config, path = load_model(
                models_root=models_root,
                run_name=spec["run_name"],
                checkpoint=spec["checkpoint"],
                source=spec["source"],
                bak_path=bak_path,
                device=args.device,
            )
        except FileNotFoundError:
            if args.skip_missing_checkpoints:
                print(f"  [SKIP] missing checkpoint: {models_root / spec['run_name'] / spec['checkpoint']}")
                continue
            raise
        print(f"  loaded: {path}")
        loaded.append((spec, model, config))
    if not loaded:
        raise ValueError("No models loaded.")

    print("=" * 100)
    print("Skeleton-L evaluation")
    print(f"benchmark_kind: {args.benchmark_kind}")
    print(f"benchmark_root: {Path(args.benchmark_root).expanduser().resolve()}")
    print(f"sample_sizes:   {parse_int_list(args.sample_sizes)}")
    print(f"threshold:      {args.threshold}")
    print(f"batch_size:     {args.batch_size}")
    print(f"device:         {args.device}")
    print("=" * 100)

    for n in parse_int_list(args.sample_sizes):
        files = discover_h5_files(args, n)
        if args.max_files_per_n is not None:
            files = files[: args.max_files_per_n]
        if not files:
            if args.skip_missing:
                print(f"[SKIP] no h5 files for n={n}")
                continue
            raise FileNotFoundError(f"No h5 files found for n={n}.")

        print(f"n={n}: files={len(files)}")
        seen_for_n = 0
        for file_idx, (h5_path, meta) in enumerate(files):
            with h5py.File(h5_path, "r") as f:
                data_all = f["data"]
                label_all = f["label"]
                file_count = int(data_all.shape[0])
                limit = file_count if args.max_datasets_per_file is None else min(file_count, args.max_datasets_per_file)
                if args.max_datasets_per_n is not None:
                    limit = min(limit, max(0, args.max_datasets_per_n - seen_for_n))
                if limit <= 0:
                    break

                graph_id = int(meta.get("graph_id", -1))
                graph_name = str(meta.get("graph_name", "random" if graph_id < 0 else f"graph_{graph_id:02d}"))
                generator = str(meta.get("generator", args.distribution if args.benchmark_kind == "random_gp" else "unknown"))
                print(
                    f"  [{file_idx + 1}/{len(files)}] graph={graph_id} generator={generator} "
                    f"{h5_path.name} datasets={limit}/{file_count}"
                )

                for start in range(0, limit, args.batch_size):
                    end = min(limit, start + args.batch_size)
                    data_batch = np.asarray(data_all[start:end], dtype=np.float32)
                    label_batch = np.asarray(label_all[start:end], dtype=int)

                    for spec, model, _ in loaded:
                        group = {
                            "method": spec["method"],
                            "run_name": spec["run_name"],
                            "checkpoint": spec["checkpoint"],
                            "checkpoint_index": spec["checkpoint_index"],
                            "benchmark_kind": args.benchmark_kind,
                            "generator": generator,
                            "graph_id": graph_id,
                            "graph_name": graph_name,
                            "sample_size": n,
                        }
                        try:
                            edge_probs_batch = decode_l_probs_batch(
                                model,
                                data_batch,
                                standardize=not args.no_standardize,
                                device=args.device,
                            )
                            for local_idx in range(edge_probs_batch.shape[0]):
                                skel_prob = skeleton_prob_from_l(edge_probs_batch[local_idx])
                                metrics = skeleton_metrics(label_batch[local_idx], skel_prob, threshold=args.threshold)
                                result_summary.add_ok(group, metrics)
                        except Exception as exc:
                            result_summary.add_error(group)
                            if args.print_errors:
                                print(f"[ERROR] {spec['method']} {h5_path} rows {start}:{end}: {exc}")

                seen_for_n += limit
                if args.max_datasets_per_n is not None and seen_for_n >= args.max_datasets_per_n:
                    break

    return result_summary.to_frame()


def save_metric_plots(summary: pd.DataFrame, plots_dir: Path, prefix: str) -> List[Path]:
    try:
        mpl_cache_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib-csnp-cache"))
        mpl_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skip plots: {repr(exc)}")
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = [
        c for c in PLOT_METRICS
        if c in summary.columns and pd.api.types.is_numeric_dtype(summary[c])
    ]
    written: List[Path] = []
    if not metric_cols:
        return written

    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        plotted = False
        for method in sorted(summary["method"].dropna().unique()):
            for n in sorted(summary["sample_size"].dropna().unique()):
                sub = summary[
                    (summary["method"] == method)
                    & (summary["sample_size"] == n)
                ].sort_values("checkpoint_index")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate skeleton learning from CSNP L_param or topo skeleton logits.")
    parser.add_argument("--methods", type=str, default="bak,ar", help="Comma-separated subset of bak,ar,topo_skeleton.")

    parser.add_argument(
        "--work_dir",
        type=str,
        default="/home/rtyen/projects/CausalStructureNeuralProcess-main/ml2_meta_causal_discovery",
    )
    parser.add_argument("--models_root", type=str, default=None, help="Optional root containing model run directories.")
    parser.add_argument(
        "--bak_path",
        type=str,
        default="ml2_meta_causal_discovery/models/causaltransformernp.py.mask_version.bak",
    )

    parser.add_argument("--bak_run_name", type=str, default="gp_4var_prob_bakL_100k")
    parser.add_argument("--bak_checkpoint", type=str, default="model_9.pt")
    parser.add_argument("--bak_checkpoints", type=str, default="")
    parser.add_argument("--bak_checkpoint_start", type=int, default=None)
    parser.add_argument("--bak_checkpoint_end", type=int, default=None)
    parser.add_argument("--bak_checkpoint_prefix", type=str, default="model_")
    parser.add_argument("--bak_checkpoint_suffix", type=str, default=".pt")
    parser.add_argument("--bak_model_source", type=str, default="bak", choices=["bak", "current"])
    parser.add_argument("--ar_run_name", type=str, default="gp_4var_prob_ar_bakL_100k")
    parser.add_argument("--ar_checkpoint", type=str, default="model_9.pt")
    parser.add_argument("--ar_checkpoints", type=str, default="")
    parser.add_argument("--ar_checkpoint_start", type=int, default=None)
    parser.add_argument("--ar_checkpoint_end", type=int, default=None)
    parser.add_argument("--ar_checkpoint_prefix", type=str, default="model_")
    parser.add_argument("--ar_checkpoint_suffix", type=str, default=".pt")
    parser.add_argument("--ar_model_source", type=str, default="current", choices=["bak", "current"])
    parser.add_argument("--topo_run_name", type=str, default="")
    parser.add_argument("--topo_checkpoint", type=str, default="model_0.pt")
    parser.add_argument("--topo_checkpoints", type=str, default="")
    parser.add_argument("--topo_checkpoint_start", type=int, default=None)
    parser.add_argument("--topo_checkpoint_end", type=int, default=None)
    parser.add_argument("--topo_checkpoint_prefix", type=str, default="model_")
    parser.add_argument("--topo_checkpoint_suffix", type=str, default=".pt")
    parser.add_argument("--topo_method_name", type=str, default="topo_skeleton")

    parser.add_argument("--benchmark_kind", type=str, default="random_gp", choices=["random_gp", "fixed_graph"])
    parser.add_argument("--benchmark_root", type=str, default="benchmark_data_4var")
    parser.add_argument("--distribution", type=str, default="csnp_gp_4var_ERL0U1")
    parser.add_argument("--fixed_generators", type=str, default="csnp_gp,rff_gaussian")
    parser.add_argument("--graph_ids", type=str, default="all")
    parser.add_argument("--sample_sizes", type=str, default="5,20,50,100,300,1000,3000")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_standardize", action="store_true")

    parser.add_argument("--max_files_per_n", type=int, default=None)
    parser.add_argument("--max_datasets_per_file", type=int, default=None)
    parser.add_argument("--max_datasets_per_n", type=int, default=None)
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--skip_missing_checkpoints", action="store_true")
    parser.add_argument("--print_errors", action="store_true")

    parser.add_argument("--results_dir", type=str, default="result")
    parser.add_argument("--summary_name", type=str, default="skeleton_L_summary.csv")
    args = parser.parse_args()

    if args.device == "cuda" and not th.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.")
        args.device = "cpu"

    df = evaluate(args)
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / args.summary_name
    args_path = results_dir / f"{out_path.stem}_args.json"
    df.to_csv(out_path, index=False)
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Wrote summary: {out_path.resolve()}")
    print(f"Wrote args:    {args_path.resolve()}")
    plot_paths = save_metric_plots(df, results_dir / "plots", out_path.stem)
    if plot_paths:
        print(f"Wrote {len(plot_paths)} plots under: {(results_dir / 'plots').resolve()}")


if __name__ == "__main__":
    main()
