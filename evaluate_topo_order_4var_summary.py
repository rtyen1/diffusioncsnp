#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate whether learned permutation/topological-order samples are valid for the
true DAG.

This script evaluates only Q/order quality. It ignores L and edge probabilities.

For each dataset and model, it samples num_order_samples orders and checks
whether each sampled root-to-leaf order is a valid topological order of the
true DAG, i.e. every true edge i -> j has position(i) < position(j).
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import re
import time
from collections import defaultdict
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch as th

from ml2_meta_causal_discovery.utils.permutations import sample_permutation


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in parse_csv_list(s)]


def maybe_standardize(x: np.ndarray, standardize: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not standardize:
        return x
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)


def remove_diag(g: np.ndarray) -> np.ndarray:
    out = np.asarray(g).copy()
    np.fill_diagonal(out, 0)
    return out


def attr_value(attrs: Dict[str, Any], key: str, default: Any) -> Any:
    v = attrs.get(key, default)
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.generic):
        return v.item()
    return v


def graph_id_from_folder(path: Path) -> int:
    match = re.match(r"graph_(\d+)__", path.name)
    return int(match.group(1)) if match else -1


def read_h5_meta(h5_path: Path, fallback: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(fallback)
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f.attrs)
        data_shape = tuple(f["data"].shape)
    meta["num_datasets"] = int(attr_value(attrs, "num_datasets", data_shape[0]))
    meta["num_samples"] = int(attr_value(attrs, "num_samples", data_shape[1]))
    meta["graph_id"] = int(attr_value(attrs, "graph_id", meta.get("graph_id", -1)))
    meta["graph_name"] = str(attr_value(attrs, "graph_name", meta.get("graph_name", "unknown")))
    meta["generator"] = str(meta.get("generator") or attr_value(attrs, "generator", "unknown"))
    return meta


def discover_h5_files(args: argparse.Namespace, sample_size: int) -> List[Tuple[Path, Dict[str, Any]]]:
    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    if args.benchmark_kind == "random_gp":
        pattern = benchmark_root / args.distribution / f"n_{sample_size}" / f"seed_{args.seed}" / "*.h5"
        fallback = {
            "benchmark": "random_gp",
            "generator": args.distribution,
            "graph_id": -1,
            "graph_name": "random_graphs",
        }
        return [(Path(p), read_h5_meta(Path(p), fallback)) for p in sorted(glob.glob(str(pattern)))]

    if args.benchmark_kind == "fixed_graph":
        graph_ids = None if args.graph_ids == "all" else set(parse_int_list(args.graph_ids))
        generators = parse_csv_list(args.fixed_generators)
        out: List[Tuple[Path, Dict[str, Any]]] = []
        for graph_dir in sorted((benchmark_root / str(args.num_nodes)).glob("graph_*")):
            if not graph_dir.is_dir():
                continue
            gid = graph_id_from_folder(graph_dir)
            if graph_ids is not None and gid not in graph_ids:
                continue
            for generator in generators:
                pattern = graph_dir / generator / f"n_{sample_size}" / f"seed_{args.seed}" / "*.h5"
                for p in sorted(glob.glob(str(pattern))):
                    fallback = {
                        "benchmark": "fixed_graph",
                        "generator": generator,
                        "graph_id": gid,
                        "graph_name": graph_dir.name.split("__", 1)[1] if "__" in graph_dir.name else graph_dir.name,
                    }
                    out.append((Path(p), read_h5_meta(Path(p), fallback)))
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
        loader = SourceFileLoader("csnp_bak_causaltransformernp_topo", str(bak_path))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load bak model file: {bak_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.CausalProbabilisticDecoder, getattr(module, "CausalProbabilisticARDecoder", None)
    raise ValueError(f"Unknown model source: {source}")


def load_model(
    *,
    csnp_work_dir: Path,
    run_name: str,
    checkpoint: str,
    source: str,
    device: str,
    bak_path: Path,
) -> Tuple[Any, Dict[str, Any], Path]:
    model_dir = csnp_work_dir / "experiments" / "causal_classification" / "models" / run_name
    config_path = model_dir / "config.json"
    model_path = model_dir / checkpoint
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    module = config.get("module", "probabilistic")
    if module == "topo_diffusion":
        from ml2_meta_causal_discovery.models.topo_order_diffusion import (
            CausalTopoOrderDiffusion,
        )

        model = CausalTopoOrderDiffusion(
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
            device=device,
            dtype=th.float32,
        ).to(device)
    else:
        CausalProbabilisticDecoder, CausalProbabilisticARDecoder = import_decoder_classes(source, bak_path)
        model_kwargs = dict(
            d_model=config["d_model"],
            emb_depth=1,
            dim_feedforward=config["dim_feedforward"],
            nhead=config["nhead"],
            dropout=0.0,
            num_layers_encoder=config["num_layers_encoder"],
            num_layers_decoder=config["num_layers_decoder"],
            num_nodes=config["num_nodes"],
            n_perm_samples=config["n_perm_samples"],
            sinkhorn_iter=config.get("sinkhorn_iter", 300),
            use_positional_encoding=config["use_positional_encoding"],
            device=device,
            dtype=th.float32,
        )
        if module == "probabilistic":
            model = CausalProbabilisticDecoder(**model_kwargs).to(device)
        elif module == "probabilistic_ar":
            if CausalProbabilisticARDecoder is None:
                raise ValueError(f"Model source {source!r} does not contain CausalProbabilisticARDecoder.")
            model = CausalProbabilisticARDecoder(
                **model_kwargs,
                num_topo_order_samples=config.get("num_topo_order_samples", 8),
                ar_hidden_dim=config.get("ar_hidden_dim", None),
            ).to(device)
        else:
            raise ValueError(f"Unsupported module for topology evaluation: {module!r}")

    try:
        state_dict = th.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = th.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, model_path


def encode_input(data: np.ndarray, device: str, standardize: bool) -> th.Tensor:
    x = maybe_standardize(data, standardize=standardize).astype(np.float32)
    return th.tensor(x, dtype=th.float32, device=device).unsqueeze(0)


def sample_ar_orders(
    *,
    model: Any,
    data: np.ndarray,
    num_order_samples: int,
    device: str,
    standardize: bool,
) -> np.ndarray:
    inputs = encode_input(data, device=device, standardize=standardize)
    with th.no_grad():
        _, q_rep, decoder_mask = model._encode_decode(target_data=inputs, mask=None)
        valid_nodes = (decoder_mask != -float("inf")) if decoder_mask is not None else None
        orders, _ = model.ar_perm_decoder.sample(q_rep, num_samples=num_order_samples, valid_nodes=valid_nodes)
    # [num_samples, batch=1, d] root-to-leaf.
    return orders[:, 0].detach().cpu().numpy().astype(int)


def sample_bak_orders(
    *,
    model: Any,
    data: np.ndarray,
    num_order_samples: int,
    device: str,
    standardize: bool,
) -> np.ndarray:
    inputs = encode_input(data, device=device, standardize=standardize)
    with th.no_grad():
        target_data = inputs.unsqueeze(-1)
        representation = model.encode(target_data=target_data, mask=None).squeeze(2)
        _, q_rep = model.decode(representation=representation, mask=None)
        p_param = model.p_param(q_rep).squeeze(-1)
        ovector = th.arange(1, model.num_nodes + 1, device=p_param.device, dtype=p_param.dtype)
        q_param = th.einsum("bn,m->bnm", p_param, ovector[: representation.size(1)])
        q_param = th.functional.F.logsigmoid(q_param)
        perm, _ = sample_permutation(
            log_alpha=q_param,
            temp=1.0,
            noise_factor=1.0,
            n_samples=num_order_samples,
            hard=True,
            n_iters=model.sinkhorn_iter,
            squeeze=False,
            device=q_param.device,
        )
        # sample_permutation returns [batch, samples, node, position].
        perm = perm.transpose(1, 0)[:, 0]  # [samples, node, position]

    orders: List[np.ndarray] = []
    for mat in perm.detach().cpu().numpy():
        # bak convention: perm[node, position] = 1, larger positions are roots.
        node_at_position = mat.argmax(axis=0)
        root_to_leaf = node_at_position[::-1]
        orders.append(root_to_leaf.astype(int))
    return np.stack(orders, axis=0)


def sample_topo_diffusion_orders(
    *,
    model: Any,
    data: np.ndarray,
    num_order_samples: int,
    device: str,
    standardize: bool,
    deterministic: bool = False,
) -> np.ndarray:
    inputs = encode_input(data, device=device, standardize=standardize)
    with th.no_grad():
        node_repr = model._encode_raw_data(inputs, mask=None)
        node_repr = node_repr.repeat_interleave(num_order_samples, dim=0)
        was_training = model.reverse_model.training
        model.reverse_model.eval()
        try:
            _, orders = model.diffusion_utils.p_sample_loop(
                node_repr,
                model.reverse_model,
                deterministic=deterministic,
            )
        finally:
            model.reverse_model.train(was_training)
    # Batched p_sample_loop returns [num_samples, d], root-to-leaf.
    return orders.detach().cpu().numpy().astype(int)


def is_valid_topological_order(adj: np.ndarray, order: np.ndarray) -> Tuple[bool, int, float]:
    adj = remove_diag(adj).astype(int)
    order = np.asarray(order, dtype=int)
    pos = np.empty(adj.shape[0], dtype=int)
    pos[order] = np.arange(len(order))
    edges = [(i, j) for i in range(adj.shape[0]) for j in range(adj.shape[1]) if adj[i, j] == 1]
    violations = sum(1 for i, j in edges if pos[i] >= pos[j])
    violation_rate = violations / len(edges) if edges else 0.0
    return violations == 0, int(violations), float(violation_rate)


def order_key(order: np.ndarray) -> Tuple[int, ...]:
    return tuple(np.asarray(order, dtype=int).tolist())


def summarize_orders(adj: np.ndarray, orders: np.ndarray) -> Dict[str, float]:
    valid_flags: List[float] = []
    violations: List[float] = []
    violation_rates: List[float] = []
    counts: Dict[Tuple[int, ...], int] = defaultdict(int)
    validity_by_order: Dict[Tuple[int, ...], float] = {}

    for order in orders:
        valid, n_viol, viol_rate = is_valid_topological_order(adj, order)
        key = order_key(order)
        counts[key] += 1
        validity_by_order[key] = float(valid)
        valid_flags.append(float(valid))
        violations.append(float(n_viol))
        violation_rates.append(float(viol_rate))

    mode_key, mode_count = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0]
    return {
        "topo_valid_rate": float(np.mean(valid_flags)),
        "mode_order_valid": validity_by_order[mode_key],
        "num_unique_orders": float(len(counts)),
        "mode_order_fraction": float(mode_count / len(orders)),
        "avg_num_violations": float(np.mean(violations)),
        "avg_violation_rate": float(np.mean(violation_rates)),
    }


class RunningSummary:
    def __init__(self, group_cols: Sequence[str]) -> None:
        self.group_cols = list(group_cols)
        self.sums: Dict[Tuple[Any, ...], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.counts: Dict[Tuple[Any, ...], int] = defaultdict(int)
        self.errors: Dict[Tuple[Any, ...], int] = defaultdict(int)

    def add_ok(self, group: Dict[str, Any], metrics: Dict[str, float]) -> None:
        key = tuple(group[c] for c in self.group_cols)
        self.counts[key] += 1
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
                self.sums[key][k] += float(v)

    def add_error(self, group: Dict[str, Any]) -> None:
        self.errors[tuple(group[c] for c in self.group_cols)] += 1

    def to_frame(self) -> pd.DataFrame:
        keys = sorted(set(self.counts) | set(self.errors))
        rows: List[Dict[str, Any]] = []
        for key in keys:
            n_ok = self.counts.get(key, 0)
            row = {col: val for col, val in zip(self.group_cols, key)}
            row["n_ok"] = n_ok
            row["n_errors"] = self.errors.get(key, 0)
            for metric, total in sorted(self.sums.get(key, {}).items()):
                row[f"mean_{metric}"] = total / n_ok if n_ok > 0 else np.nan
            rows.append(row)
        return pd.DataFrame(rows)


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    device = "cuda" if args.device == "auto" and th.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    csnp_work_dir = Path(args.csnp_work_dir).expanduser().resolve()
    bak_path = Path(args.bak_model_file).expanduser().resolve()
    sample_sizes = parse_int_list(args.sample_sizes)

    requested_methods = set(parse_csv_list(args.methods))
    valid_methods = {"bak", "ar", "topo"}
    unknown_methods = requested_methods - valid_methods
    if unknown_methods:
        raise ValueError(f"Unknown methods: {sorted(unknown_methods)}. Use any of: {sorted(valid_methods)}")

    model_specs = []
    if "bak" in requested_methods:
        model_specs.append(
            {
                "method": args.bak_method_name,
                "run_name": args.bak_run_name,
                "checkpoint": args.bak_checkpoint,
                "source": "bak",
                "kind": "bak",
            }
        )
    if "ar" in requested_methods:
        model_specs.append(
            {
                "method": args.ar_method_name,
                "run_name": args.ar_run_name,
                "checkpoint": args.ar_checkpoint,
                "source": "current",
                "kind": "ar",
            }
        )
    if "topo" in requested_methods:
        if not args.topo_run_name:
            raise ValueError("Use --topo_run_name when --methods includes topo.")
        topo_eval_modes = set(parse_csv_list(args.topo_eval_modes))
        valid_topo_modes = {"sample", "deterministic"}
        unknown_topo_modes = topo_eval_modes - valid_topo_modes
        if unknown_topo_modes:
            raise ValueError(
                f"Unknown topo_eval_modes: {sorted(unknown_topo_modes)}. "
                f"Use any of: {sorted(valid_topo_modes)}"
            )
        if "sample" in topo_eval_modes:
            model_specs.append(
                {
                    "method": args.topo_method_name,
                    "run_name": args.topo_run_name,
                    "checkpoint": args.topo_checkpoint,
                    "source": "current",
                    "kind": "topo_diffusion_sample",
                    "variant_suffix": "sample",
                }
            )
        if "deterministic" in topo_eval_modes:
            model_specs.append(
                {
                    "method": f"{args.topo_method_name}-det",
                    "run_name": args.topo_run_name,
                    "checkpoint": args.topo_checkpoint,
                    "source": "current",
                    "kind": "topo_diffusion_deterministic",
                    "variant_suffix": "deterministic",
                }
            )
    if not model_specs:
        raise ValueError("No methods selected.")

    loaded_models = []
    print("=" * 100)
    print("Topology-order evaluation")
    print(f"benchmark_kind: {args.benchmark_kind}")
    print(f"benchmark_root: {Path(args.benchmark_root).expanduser().resolve()}")
    print(f"sample_sizes:   {sample_sizes}")
    print(f"num_order_samples: {args.num_order_samples}")
    print(f"device:         {device}")
    print("=" * 100)
    for spec in model_specs:
        print(f"Loading {spec['method']}: {spec['run_name']}/{spec['checkpoint']} source={spec['source']}")
        model, _, model_path = load_model(
            csnp_work_dir=csnp_work_dir,
            run_name=spec["run_name"],
            checkpoint=spec["checkpoint"],
            source=spec["source"],
            device=device,
            bak_path=bak_path,
        )
        variant = f"{spec['run_name']}_{spec['checkpoint']}"
        if "variant_suffix" in spec:
            variant = f"{variant}_{spec['variant_suffix']}"
        variant = f"{variant}_orders{args.num_order_samples}"
        loaded_models.append((spec["method"], spec["kind"], variant, model))
        print(f"  loaded: {model_path}")

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
                    for method, kind, variant, model in loaded_models:
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
                            if kind == "ar":
                                orders = sample_ar_orders(
                                    model=model,
                                    data=data,
                                    num_order_samples=args.num_order_samples,
                                    device=device,
                                    standardize=args.standardize,
                                )
                            elif kind in {"topo_diffusion_sample", "topo_diffusion_deterministic"}:
                                orders = sample_topo_diffusion_orders(
                                    model=model,
                                    data=data,
                                    num_order_samples=args.num_order_samples,
                                    device=device,
                                    standardize=args.standardize,
                                    deterministic=(kind == "topo_diffusion_deterministic"),
                                )
                            else:
                                orders = sample_bak_orders(
                                    model=model,
                                    data=data,
                                    num_order_samples=args.num_order_samples,
                                    device=device,
                                    standardize=args.standardize,
                                )
                            summary.add_ok(group, summarize_orders(true_dag, orders))
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
    parser = argparse.ArgumentParser(description="Evaluate learned topological-order samples for 4var benchmarks.")
    parser.add_argument("--csnp_work_dir", type=str, default="/home/rtyen/projects/CausalStructureNeuralProcess-main/ml2_meta_causal_discovery")
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
    parser.add_argument("--methods", type=str, default="bak,ar", help="Comma-separated subset of bak,ar,topo.")

    parser.add_argument("--bak_run_name", type=str, default="gp_4var_prob_bakL_100k")
    parser.add_argument("--bak_checkpoint", type=str, default="model_9.pt")
    parser.add_argument("--bak_method_name", type=str, default="CSNP-bakL-model9")
    parser.add_argument("--ar_run_name", type=str, default="gp_4var_prob_ar_bakL_100k_continue13_more20")
    parser.add_argument("--ar_checkpoint", type=str, default="model_19.pt")
    parser.add_argument("--ar_method_name", type=str, default="AR-latest-model19")
    parser.add_argument("--topo_run_name", type=str, default="")
    parser.add_argument("--topo_checkpoint", type=str, default="model_11.pt")
    parser.add_argument("--topo_method_name", type=str, default="GPL-topo-diffusion")
    parser.add_argument("--topo_eval_modes", type=str, default="sample", help="Comma-separated subset of sample,deterministic.")
    parser.add_argument("--bak_model_file", type=str, default="ml2_meta_causal_discovery/models/causaltransformernp.py.mask_version.bak")

    parser.add_argument("--results_dir", type=str, default="benchmark_results_4var_topo_order")
    parser.add_argument("--summary_name", type=str, default="summary_topo_order_4var.csv")
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
    print(f"Wrote topology-order summary: {out_path}")


if __name__ == "__main__":
    main()
