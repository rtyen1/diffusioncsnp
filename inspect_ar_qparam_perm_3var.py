#!/usr/bin/env python3
"""Inspect AR permutation decoder outputs for 3-variable datasets.

This is the autoregressive counterpart of inspect_bak_qparam_perm_3var.py.
The AR model has no single d x d Q_param matrix. Instead it defines

    P(order | X) = prod_t P(q_t | q_<t, X)

where order is root-to-leaf. This script records sampled orders, their
bak-compatible permutation matrices, the resulting masks, L, and graph
probabilities.
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch as th

from ml2_meta_causal_discovery.models.causaltransformernp import CausalProbabilisticARDecoder
from ml2_meta_causal_discovery.utils.topological_orders import orders_to_bak_permutation


EDGE_ORDER_3VAR: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
]


def json_dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def finite_json_list(x: np.ndarray) -> Any:
    arr = np.asarray(x)
    if arr.ndim == 0:
        value = float(arr)
        return value if math.isfinite(value) else None
    return [finite_json_list(v) for v in arr]


def parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def matrix_to_edges(g: np.ndarray) -> str:
    g = np.asarray(g).astype(int).copy()
    np.fill_diagonal(g, 0)
    edges = []
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if i != j and g[i, j] == 1:
                edges.append(f"{i}->{j}")
    return ", ".join(edges) if edges else "(none)"


def flat_graph_columns(prefix: str, g: np.ndarray) -> Dict[str, int]:
    g = np.asarray(g).astype(int).copy()
    np.fill_diagonal(g, 0)
    return {f"{prefix}_{i}to{j}": int(g[i, j]) for i, j in EDGE_ORDER_3VAR}


def flat_matrix_columns(prefix: str, g: np.ndarray) -> Dict[str, float]:
    g = np.asarray(g, dtype=float)
    return {f"{prefix}_{i}to{j}": float(g[i, j]) for i, j in EDGE_ORDER_3VAR}


def standardize_batch(x: np.ndarray, standardize: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if not standardize:
        return x
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)


def rank_to_perm_matrix(rank: Sequence[int]) -> List[List[int]]:
    d = len(rank)
    mat = np.eye(d, dtype=int)[np.asarray(rank, dtype=int)]
    return mat.tolist()


def order_to_rank(order: Sequence[int]) -> Tuple[int, ...]:
    """Convert root-to-leaf order to bak node->position ranks."""
    d = len(order)
    rank = [0] * d
    for t, node in enumerate(order):
        rank[int(node)] = d - 1 - t
    return tuple(rank)


def iter_batches(n_items: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n_items, batch_size):
        yield start, min(start + batch_size, n_items)


def attr_value(attrs: Dict[str, Any], key: str, default: Any) -> Any:
    v = attrs.get(key, default)
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.generic):
        return v.item()
    return v


def read_h5_metadata(f: h5py.File, fallback_num_samples: int) -> Dict[str, Any]:
    attrs = dict(f.attrs)
    return {
        "graph_id": int(attr_value(attrs, "graph_id", -1)),
        "graph_name": str(attr_value(attrs, "graph_name", "unknown")),
        "graph_bitstring": str(attr_value(attrs, "graph_bitstring", "")),
        "num_samples": int(attr_value(attrs, "num_samples", fallback_num_samples)),
        "mechanism": str(attr_value(attrs, "mechanism", "")),
        "noise": str(attr_value(attrs, "noise", "")),
        "seed": int(attr_value(attrs, "seed", -1)),
    }


def load_model(args: argparse.Namespace, device: str) -> Tuple[Any, Dict[str, Any], Path]:
    work_dir = Path(args.work_dir).expanduser().resolve()
    model_dir = work_dir / "experiments" / "causal_classification" / "models" / args.run_name
    config_path = model_dir / "config.json"
    model_path = model_dir / args.model_checkpoint

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    module = config.get("module", "")
    if module != "probabilistic_ar":
        raise ValueError(f"Expected config module='probabilistic_ar', got {module!r}.")

    model = CausalProbabilisticARDecoder(
        d_model=config["d_model"],
        emb_depth=1,
        dim_feedforward=config["dim_feedforward"],
        nhead=config["nhead"],
        dropout=0.0,
        num_layers_encoder=config["num_layers_encoder"],
        num_layers_decoder=config["num_layers_decoder"],
        num_nodes=config["num_nodes"],
        n_perm_samples=config["n_perm_samples"],
        sinkhorn_iter=config["sinkhorn_iter"],
        use_positional_encoding=config["use_positional_encoding"],
        num_topo_order_samples=config.get("num_topo_order_samples", 8),
        ar_hidden_dim=config.get("ar_hidden_dim", None),
        device=device,
        dtype=th.float32,
    ).to(device)

    try:
        state_dict = th.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = th.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, model_path


def resolve_h5_files(args: argparse.Namespace) -> List[Path]:
    if args.source == "avici_benchmark":
        benchmark_root = Path(args.benchmark_root).expanduser().resolve()
        graph_ids = parse_int_list(args.graph_ids)
        sample_sizes = parse_int_list(args.sample_sizes)
        files = []
        for graph_id in graph_ids:
            for num_samples in sample_sizes:
                pattern = (
                    benchmark_root
                    / "3"
                    / f"graph_{graph_id:02d}__*"
                    / args.mechanism
                    / args.noise
                    / f"n_{num_samples}"
                    / f"seed_{args.benchmark_seed}"
                    / f"benchmark_3var_numdatasets_{args.num_datasets}.h5"
                )
                matches = sorted(Path(p) for p in glob.glob(str(pattern)))
                if len(matches) != 1:
                    if args.skip_missing:
                        print(f"[SKIP] expected one h5, found {len(matches)}: {pattern}")
                        continue
                    raise FileNotFoundError(f"Expected one h5, found {len(matches)}: {pattern}")
                files.append(matches[0])
    elif args.data_path:
        path = Path(args.data_path).expanduser().resolve()
        if path.is_file():
            files = [path]
        else:
            files = sorted(path.glob("*.hdf5")) + sorted(path.glob("*.h5"))
    else:
        work_dir = Path(args.work_dir).expanduser().resolve()
        data_dir = work_dir / "datasets" / "data" / "synth_training_data" / args.data_file / args.split
        files = sorted(data_dir.glob("*.hdf5")) + sorted(data_dir.glob("*.h5"))

    h5_indices = parse_int_list(args.h5_indices)
    if h5_indices:
        files = [p for p in files if any(p.stem.endswith(f"_{idx}") for idx in h5_indices)]
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError("No h5/hdf5 files found for the requested dataset.")
    return files


def all_permutation_orders(num_nodes: int, device: str, max_exact_orders: int) -> Optional[th.Tensor]:
    n_perm = math.factorial(num_nodes)
    if max_exact_orders <= 0 or n_perm > max_exact_orders:
        return None
    orders = list(itertools.permutations(range(num_nodes)))
    return th.tensor(orders, dtype=th.long, device=device).unsqueeze(1)


def ar_step_details(model: Any, q_rep: th.Tensor, orders: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    """
    Return chosen step logits and probabilities for teacher-forced orders.

    Args:
        q_rep: [batch_size, d, d_model]
        orders: [num_orders, batch_size, d], root-to-leaf

    Returns:
        chosen_logits, chosen_probs: both [num_orders, batch_size, d]
    """
    num_orders, batch_size, order_len = orders.shape
    q_rep_rep = q_rep.repeat_interleave(num_orders, dim=0)
    flat_orders = orders.transpose(0, 1).contiguous().view(batch_size * num_orders, order_len)
    selected = th.zeros((batch_size * num_orders, q_rep.size(1)), dtype=th.bool, device=q_rep.device)
    state = model.ar_perm_decoder._initial_state(q_rep_rep)
    batch_idx = th.arange(batch_size * num_orders, device=q_rep.device)
    chosen_logits = []
    chosen_probs = []

    for t in range(order_len):
        logits = model.ar_perm_decoder._step_logits(q_rep_rep, state, selected)
        probs = th.softmax(logits.float(), dim=-1).to(q_rep.dtype)
        chosen = flat_orders[:, t].long()
        chosen_logits.append(logits[batch_idx, chosen])
        chosen_probs.append(probs[batch_idx, chosen])
        selected = selected.clone()
        selected[batch_idx, chosen] = True
        state = model.ar_perm_decoder.gru(q_rep_rep[batch_idx, chosen], state)

    logits_out = th.stack(chosen_logits, dim=-1).view(batch_size, num_orders, order_len).transpose(0, 1)
    probs_out = th.stack(chosen_probs, dim=-1).view(batch_size, num_orders, order_len).transpose(0, 1)
    return logits_out, probs_out


def inspect_ar_order_and_l(
    model: Any,
    inputs: th.Tensor,
    num_perm_samples: int,
    max_exact_orders: int,
    mask: Optional[th.Tensor] = None,
) -> Dict[str, np.ndarray]:
    l_param, q_rep, decoder_mask = model._encode_decode(target_data=inputs, mask=mask)
    edge_probs = th.sigmoid(l_param)
    valid_nodes = (decoder_mask != -float("inf")) if decoder_mask is not None else None

    orders, log_p_q = model.ar_perm_decoder.sample(
        q_rep,
        num_samples=num_perm_samples,
        valid_nodes=valid_nodes,
    )
    perm = orders_to_bak_permutation(orders, num_nodes=q_rep.size(1), dtype=edge_probs.dtype)
    perm_inv = perm.transpose(2, 3)
    tri_mask = model._lower_mask(edge_probs.size(-1), perm.device, perm.dtype)
    sampled_masks = th.einsum("sbij,jk,sbkl->sbil", perm, tri_mask, perm_inv)
    sampled_probs = edge_probs[None] * sampled_masks
    mean_prob = sampled_probs.mean(dim=0)
    sampled_step_logits, sampled_step_probs = ar_step_details(model, q_rep, orders)

    exact_orders_1b = all_permutation_orders(q_rep.size(1), str(q_rep.device), max_exact_orders)
    exact_orders = None
    exact_log_p_q = None
    exact_step_logits = None
    exact_step_probs = None
    if exact_orders_1b is not None:
        exact_orders = exact_orders_1b.expand(-1, q_rep.size(0), -1).contiguous()
        exact_log_p_q = model.ar_perm_decoder.log_prob(q_rep, exact_orders, valid_nodes=valid_nodes)
        exact_step_logits, exact_step_probs = ar_step_details(model, q_rep, exact_orders)

    return {
        "l_param": l_param.detach().cpu().numpy(),
        "edge_probs": edge_probs.detach().cpu().numpy(),
        "mean_prob": mean_prob.detach().cpu().numpy(),
        "orders": orders.detach().cpu().numpy(),
        "log_p_q": log_p_q.detach().cpu().numpy(),
        "perm": perm.detach().cpu().numpy(),
        "sampled_masks": sampled_masks.detach().cpu().numpy(),
        "sampled_probs": sampled_probs.detach().cpu().numpy(),
        "sampled_step_logits": sampled_step_logits.detach().cpu().numpy(),
        "sampled_step_probs": sampled_step_probs.detach().cpu().numpy(),
        "exact_orders": None if exact_orders is None else exact_orders.detach().cpu().numpy(),
        "exact_log_p_q": None if exact_log_p_q is None else exact_log_p_q.detach().cpu().numpy(),
        "exact_step_logits": None if exact_step_logits is None else exact_step_logits.detach().cpu().numpy(),
        "exact_step_probs": None if exact_step_probs is None else exact_step_probs.detach().cpu().numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record AR permutation decoder sampled orders, L, masks, and probabilities for 3-variable datasets."
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="/home/rtyen/projects/CausalStructureNeuralProcess-main/ml2_meta_causal_discovery",
    )
    parser.add_argument("--run_name", type=str, default="gp_3var_prob_ar_bakL_100k")
    parser.add_argument("--model_checkpoint", type=str, default="model_9.pt")
    parser.add_argument(
        "--source",
        type=str,
        default="csnp_synth",
        choices=["csnp_synth", "avici_benchmark"],
    )
    parser.add_argument("--data_file", type=str, default="gp_3var_ERL0U1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--benchmark_root", type=str, default="/home/rtyen/projects/avici-main/benchmark_data")
    parser.add_argument("--graph_ids", type=str, default="0,1,15,16,17,24")
    parser.add_argument("--sample_sizes", type=str, default="50,100,200,500,1000,3000")
    parser.add_argument("--num_datasets", type=int, default=100)
    parser.add_argument("--benchmark_seed", type=int, default=42)
    parser.add_argument("--mechanism", type=str, default="linear")
    parser.add_argument("--noise", type=str, default="gaussian")
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--h5_indices", type=str, default="")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_datasets_per_file", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--max_node_attention_batches",
        type=int,
        default=60000,
        help="Cap batch_size * (num_observations + 1) to avoid CUDA attention kernel launch limits.",
    )
    parser.add_argument("--num_perm_samples", type=int, default=None)
    parser.add_argument(
        "--max_exact_orders",
        type=int,
        default=720,
        help="Enumerate exact AR probabilities if num_nodes! <= this value. Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no_standardize", action="store_true")
    parser.add_argument("--output_dir", type=str, default="ar_qparam_perm_results")
    parser.add_argument("--output_prefix", type=str, default="")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"
    else:
        device = args.device

    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if device == "cuda":
        th.cuda.manual_seed_all(args.seed)

    model, config, model_path = load_model(args, device=device)
    num_perm_samples = int(args.num_perm_samples or config["n_perm_samples"])
    h5_files = resolve_h5_files(args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"{args.run_name}_{args.model_checkpoint}_{args.data_file}_{args.split}_perm{num_perm_samples}"
    if args.source == "avici_benchmark" and not args.output_prefix:
        prefix = (
            f"{args.run_name}_{args.model_checkpoint}_avici_"
            f"{args.mechanism}_{args.noise}_seed{args.benchmark_seed}_perm{num_perm_samples}"
        )
    dataset_csv = output_dir / f"{prefix}_datasets.csv"
    order_counts_csv = output_dir / f"{prefix}_order_counts.csv"
    exact_orders_csv = output_dir / f"{prefix}_exact_order_probs.csv"
    global_counts_csv = output_dir / f"{prefix}_global_order_counts.csv"

    dataset_fieldnames = [
        "h5_path",
        "h5_file",
        "graph_id",
        "graph_name",
        "graph_bitstring",
        "num_samples",
        "mechanism",
        "noise",
        "seed",
        "data_idx",
        "true_graph_json",
        "true_edges",
        *[f"true_{i}to{j}" for i, j in EDGE_ORDER_3VAR],
        "ar_q_param_note",
        "l_param_json",
        "edge_probs_json",
        "mean_prob_json",
        *[f"mean_prob_{i}to{j}" for i, j in EDGE_ORDER_3VAR],
        "num_perm_samples",
        "num_unique_orders",
        "exact_orders_enumerated",
        "model_path",
    ]
    order_fieldnames = [
        "h5_path",
        "h5_file",
        "graph_id",
        "graph_name",
        "graph_bitstring",
        "num_samples",
        "mechanism",
        "noise",
        "seed",
        "data_idx",
        "true_graph_json",
        "true_edges",
        *[f"true_{i}to{j}" for i, j in EDGE_ORDER_3VAR],
        "root_to_leaf_json",
        "leaf_to_root_json",
        "perm_rank_node_to_position_json",
        "perm_matrix_json",
        "perm_mask_json",
        "perm_prob_json",
        "root_node",
        "leaf_node",
        "count",
        "frequency",
        "mean_log_p_order_sampled",
        "mean_p_order_sampled",
        "step_chosen_logits_json",
        "step_chosen_probs_json",
        "num_perm_samples",
    ]
    exact_fieldnames = [
        "h5_path",
        "h5_file",
        "graph_id",
        "graph_name",
        "graph_bitstring",
        "num_samples",
        "mechanism",
        "noise",
        "seed",
        "data_idx",
        "true_graph_json",
        "true_edges",
        *[f"true_{i}to{j}" for i, j in EDGE_ORDER_3VAR],
        "root_to_leaf_json",
        "leaf_to_root_json",
        "perm_rank_node_to_position_json",
        "log_p_order",
        "p_order",
        "step_chosen_logits_json",
        "step_chosen_probs_json",
    ]
    global_fieldnames = [
        "true_graph_json",
        "true_edges",
        "root_to_leaf_json",
        "leaf_to_root_json",
        "perm_rank_node_to_position_json",
        "root_node",
        "leaf_node",
        "count",
    ]

    global_counter: Counter[Tuple[str, Tuple[int, ...]]] = Counter()

    print("=" * 100)
    print("Inspecting AR permutation decoder sampled orders and bak-style L/mask")
    print(f"model:             {model_path}")
    print(f"device:            {device}")
    print(f"num_perm_samples:  {num_perm_samples}")
    print(f"max_exact_orders:  {args.max_exact_orders}")
    print(f"h5 files:          {len(h5_files)}")
    print(f"dataset csv:       {dataset_csv}")
    print(f"order count csv:   {order_counts_csv}")
    print(f"exact order csv:   {exact_orders_csv}")
    print("=" * 100)

    with open(dataset_csv, "w", newline="", encoding="utf-8") as f_dataset, open(
        order_counts_csv, "w", newline="", encoding="utf-8"
    ) as f_order, open(exact_orders_csv, "w", newline="", encoding="utf-8") as f_exact:
        dataset_writer = csv.DictWriter(f_dataset, fieldnames=dataset_fieldnames)
        order_writer = csv.DictWriter(f_order, fieldnames=order_fieldnames)
        exact_writer = csv.DictWriter(f_exact, fieldnames=exact_fieldnames)
        dataset_writer.writeheader()
        order_writer.writeheader()
        exact_writer.writeheader()

        for h5_file_idx, h5_path in enumerate(h5_files):
            with h5py.File(h5_path, "r") as f:
                n_in_file = int(f["data"].shape[0])
                n_observations = int(f["data"].shape[1])
                h5_meta = read_h5_metadata(f, fallback_num_samples=int(f["data"].shape[1]))
                limit = n_in_file if args.max_datasets_per_file is None else min(n_in_file, args.max_datasets_per_file)
                effective_batch_size = int(args.batch_size)
                if args.max_node_attention_batches and args.max_node_attention_batches > 0:
                    effective_batch_size = min(
                        effective_batch_size,
                        max(1, int(args.max_node_attention_batches) // max(1, n_observations + 1)),
                    )

                print(
                    f"[{h5_file_idx + 1}/{len(h5_files)}] {h5_path} "
                    f"datasets={limit}/{n_in_file} batch={effective_batch_size}"
                )

                for start, end in iter_batches(limit, effective_batch_size):
                    data = np.asarray(f["data"][start:end], dtype=np.float32)
                    labels = np.asarray(f["label"][start:end], dtype=int)
                    data = standardize_batch(data, standardize=not args.no_standardize)
                    inputs = th.tensor(data, dtype=th.float32, device=device)

                    with th.no_grad():
                        out = inspect_ar_order_and_l(
                            model=model,
                            inputs=inputs,
                            num_perm_samples=num_perm_samples,
                            max_exact_orders=args.max_exact_orders,
                        )

                    orders = out["orders"]

                    for local_idx in range(end - start):
                        data_idx = start + local_idx
                        true_graph = labels[local_idx].astype(int)
                        np.fill_diagonal(true_graph, 0)
                        true_graph_json = json_dumps(true_graph.tolist())
                        true_edges = matrix_to_edges(true_graph)
                        true_flat = flat_graph_columns("true", true_graph)

                        order_samples = [tuple(int(x) for x in orders[s, local_idx].tolist()) for s in range(num_perm_samples)]
                        counter = Counter(order_samples)
                        exact_enumerated = out["exact_orders"] is not None

                        dataset_writer.writerow(
                            {
                                "h5_path": str(h5_path),
                                "h5_file": h5_path.name,
                                **h5_meta,
                                "data_idx": data_idx,
                                "true_graph_json": true_graph_json,
                                "true_edges": true_edges,
                                **true_flat,
                                "ar_q_param_note": "No single Q_param; AR uses conditional step probabilities P(q_t|q_<t,X).",
                                "l_param_json": json_dumps(out["l_param"][local_idx].tolist()),
                                "edge_probs_json": json_dumps(out["edge_probs"][local_idx].tolist()),
                                "mean_prob_json": json_dumps(out["mean_prob"][local_idx].tolist()),
                                **flat_matrix_columns("mean_prob", out["mean_prob"][local_idx]),
                                "num_perm_samples": num_perm_samples,
                                "num_unique_orders": len(counter),
                                "exact_orders_enumerated": bool(exact_enumerated),
                                "model_path": str(model_path),
                            }
                        )

                        for order, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
                            rank = order_to_rank(order)
                            leaf_to_root = list(reversed(order))
                            global_counter[(true_graph_json, order)] += int(count)
                            sample_indices = [i for i, x in enumerate(order_samples) if x == order]
                            first_sample_idx = sample_indices[0]
                            log_vals = out["log_p_q"][sample_indices, local_idx]
                            mean_log_p = float(np.mean(log_vals))
                            mean_p = float(np.mean(np.exp(log_vals)))
                            order_writer.writerow(
                                {
                                    "h5_path": str(h5_path),
                                    "h5_file": h5_path.name,
                                    **h5_meta,
                                    "data_idx": data_idx,
                                    "true_graph_json": true_graph_json,
                                    "true_edges": true_edges,
                                    **true_flat,
                                    "root_to_leaf_json": json_dumps(list(order)),
                                    "leaf_to_root_json": json_dumps(leaf_to_root),
                                    "perm_rank_node_to_position_json": json_dumps(list(rank)),
                                    "perm_matrix_json": json_dumps(rank_to_perm_matrix(rank)),
                                    "perm_mask_json": json_dumps(out["sampled_masks"][first_sample_idx, local_idx].astype(int).tolist()),
                                    "perm_prob_json": json_dumps(out["sampled_probs"][first_sample_idx, local_idx].tolist()),
                                    "root_node": int(order[0]),
                                    "leaf_node": int(order[-1]),
                                    "count": int(count),
                                    "frequency": float(count) / float(num_perm_samples),
                                    "mean_log_p_order_sampled": mean_log_p,
                                    "mean_p_order_sampled": mean_p,
                                    "step_chosen_logits_json": json_dumps(
                                        finite_json_list(out["sampled_step_logits"][first_sample_idx, local_idx])
                                    ),
                                    "step_chosen_probs_json": json_dumps(
                                        finite_json_list(out["sampled_step_probs"][first_sample_idx, local_idx])
                                    ),
                                    "num_perm_samples": num_perm_samples,
                                }
                            )

                        if exact_enumerated:
                            exact_orders = out["exact_orders"]
                            for exact_idx in range(exact_orders.shape[0]):
                                order = tuple(int(x) for x in exact_orders[exact_idx, local_idx].tolist())
                                rank = order_to_rank(order)
                                leaf_to_root = list(reversed(order))
                                log_p_order = float(out["exact_log_p_q"][exact_idx, local_idx])
                                exact_writer.writerow(
                                    {
                                        "h5_path": str(h5_path),
                                        "h5_file": h5_path.name,
                                        **h5_meta,
                                        "data_idx": data_idx,
                                        "true_graph_json": true_graph_json,
                                        "true_edges": true_edges,
                                        **true_flat,
                                        "root_to_leaf_json": json_dumps(list(order)),
                                        "leaf_to_root_json": json_dumps(leaf_to_root),
                                        "perm_rank_node_to_position_json": json_dumps(list(rank)),
                                        "log_p_order": log_p_order,
                                        "p_order": float(math.exp(log_p_order)),
                                        "step_chosen_logits_json": json_dumps(
                                            finite_json_list(out["exact_step_logits"][exact_idx, local_idx])
                                        ),
                                        "step_chosen_probs_json": json_dumps(
                                            finite_json_list(out["exact_step_probs"][exact_idx, local_idx])
                                        ),
                                    }
                                )

    with open(global_counts_csv, "w", newline="", encoding="utf-8") as f_global:
        writer = csv.DictWriter(f_global, fieldnames=global_fieldnames)
        writer.writeheader()
        for (true_graph_json, order), count in sorted(global_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            true_graph = np.asarray(json.loads(true_graph_json), dtype=int)
            rank = order_to_rank(order)
            leaf_to_root = list(reversed(order))
            writer.writerow(
                {
                    "true_graph_json": true_graph_json,
                    "true_edges": matrix_to_edges(true_graph),
                    "root_to_leaf_json": json_dumps(list(order)),
                    "leaf_to_root_json": json_dumps(leaf_to_root),
                    "perm_rank_node_to_position_json": json_dumps(list(rank)),
                    "root_node": int(order[0]),
                    "leaf_node": int(order[-1]),
                    "count": int(count),
                }
            )

    print("=" * 100)
    print("Done")
    print(f"Wrote: {dataset_csv}")
    print(f"Wrote: {order_counts_csv}")
    print(f"Wrote: {exact_orders_csv}")
    print(f"Wrote: {global_counts_csv}")


if __name__ == "__main__":
    main()
