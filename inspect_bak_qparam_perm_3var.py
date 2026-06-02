#!/usr/bin/env python3
"""Inspect Q_param and sampled permutations for the bak probabilistic decoder.

This script loads CausalProbabilisticDecoder from
ml2_meta_causal_discovery/models/causaltransformernp.py.mask_version.bak without
renaming files, then reproduces the bak forward path up to Q_param and perm.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.machinery
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch as th

from ml2_meta_causal_discovery.utils.permutations import sample_permutation


EDGE_ORDER_3VAR: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
]


def json_dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


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


def load_bak_decoder_class(bak_model_file: Path) -> Any:
    loader = importlib.machinery.SourceFileLoader("causaltransformernp_bak", str(bak_model_file))
    module = loader.load_module()
    return module.CausalProbabilisticDecoder


def load_model(args: argparse.Namespace, device: str) -> Tuple[Any, Dict[str, Any], Path]:
    work_dir = Path(args.work_dir).expanduser().resolve()
    model_dir = work_dir / "experiments" / "causal_classification" / "models" / args.run_name
    config_path = model_dir / "config.json"
    model_path = model_dir / args.model_checkpoint
    bak_model_file = Path(args.bak_model_file).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if not bak_model_file.exists():
        raise FileNotFoundError(f"Bak model file not found: {bak_model_file}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    Decoder = load_bak_decoder_class(bak_model_file)
    model = Decoder(
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


def inspect_qparam_and_perm(
    model: Any,
    inputs: th.Tensor,
    num_perm_samples: int,
    mask: Optional[th.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Reproduce bak forward up to Q_param and sampled perm."""
    if inputs.dim() == 3:
        target_data = inputs.unsqueeze(-1)
    else:
        target_data = inputs

    representation = model.encode(target_data=target_data, mask=mask)
    representation = representation.squeeze(2)

    if mask is not None:
        decoder_mask = mask[:, 0, :]
    else:
        decoder_mask = None

    l_param, q_rep = model.decode(representation=representation, mask=decoder_mask)
    p_param = model.p_param(q_rep).squeeze(-1)
    ovector = th.arange(
        1,
        model.num_nodes + 1,
        device=p_param.device,
        dtype=p_param.dtype,
    )
    q_param_raw = th.einsum(
        "bn,m->bnm",
        p_param,
        ovector[: representation.size(1)],
    )
    q_param_log_alpha = th.functional.F.logsigmoid(q_param_raw)

    if decoder_mask is not None:
        q_mask = decoder_mask.unsqueeze(1) + decoder_mask.unsqueeze(2)
        q_mask = q_mask * (1 - th.eye(q_mask.size(-1), device=q_mask.device))
        q_param_log_alpha = q_param_log_alpha + q_mask

    old_n_perm_samples = getattr(model, "n_perm_samples", None)
    model.n_perm_samples = int(num_perm_samples)
    perm, _ = sample_permutation(
        log_alpha=q_param_log_alpha,
        temp=1.0,
        noise_factor=1.0,
        n_samples=model.n_perm_samples,
        hard=True,
        n_iters=model.sinkhorn_iter,
        squeeze=False,
        device=q_param_log_alpha.device,
    )
    if old_n_perm_samples is not None:
        model.n_perm_samples = old_n_perm_samples

    # sample_permutation returns [batch, n_samples, d, d]; bak forward transposes it.
    perm = perm.transpose(1, 0)
    perm_inv = perm.transpose(2, 3)
    tri_mask = th.tril(
        th.ones(
            (model.num_nodes, model.num_nodes),
            device=perm.device,
            dtype=perm.dtype,
        ),
        diagonal=-1,
    )
    tri_mask = tri_mask[: representation.size(1), : representation.size(1)]
    all_masks = th.einsum(
        "sbij,jk,sbkl->sbil",
        perm,
        tri_mask,
        perm_inv,
    )
    edge_probs = th.sigmoid(l_param)
    all_probs = edge_probs[None] * all_masks
    mean_prob = all_probs.mean(dim=0)

    return {
        "l_param": l_param.detach().cpu().numpy(),
        "edge_probs": edge_probs.detach().cpu().numpy(),
        "mean_prob": mean_prob.detach().cpu().numpy(),
        "all_masks": all_masks.detach().cpu().numpy(),
        "all_probs": all_probs.detach().cpu().numpy(),
        "p_param": p_param.detach().cpu().numpy(),
        "q_param_raw": q_param_raw.detach().cpu().numpy(),
        "q_param_log_alpha": q_param_log_alpha.detach().cpu().numpy(),
        "perm": perm.detach().cpu().numpy(),
    }


def rank_to_perm_matrix(rank: Sequence[int]) -> List[List[int]]:
    d = len(rank)
    mat = np.eye(d, dtype=int)[np.asarray(rank, dtype=int)]
    return mat.tolist()


def iter_batches(n_items: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n_items, batch_size):
        yield start, min(start + batch_size, n_items)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record bak Q_param and sampled permutation counts for 3-variable datasets."
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="/home/rtyen/projects/CausalStructureNeuralProcess-main/ml2_meta_causal_discovery",
    )
    parser.add_argument(
        "--bak_model_file",
        type=str,
        default="/home/rtyen/projects/CausalStructureNeuralProcess-main/ml2_meta_causal_discovery/models/causaltransformernp.py.mask_version.bak",
    )
    parser.add_argument("--run_name", type=str, default="gp_3var_prob_rerun_100k")
    parser.add_argument("--model_checkpoint", type=str, default="model_8.pt")
    parser.add_argument(
        "--source",
        type=str,
        default="csnp_synth",
        choices=["csnp_synth", "avici_benchmark"],
        help="csnp_synth reads work_dir/datasets/data/synth_training_data; avici_benchmark reads benchmark_data.",
    )
    parser.add_argument("--data_file", type=str, default="gp_3var_ERL0U1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument(
        "--benchmark_root",
        type=str,
        default="/home/rtyen/projects/avici-main/benchmark_data",
    )
    parser.add_argument("--graph_ids", type=str, default="0,1,15,16,17,24")
    parser.add_argument("--sample_sizes", type=str, default="50,100,200,500,1000,3000")
    parser.add_argument("--num_datasets", type=int, default=100)
    parser.add_argument("--benchmark_seed", type=int, default=42)
    parser.add_argument("--mechanism", type=str, default="linear")
    parser.add_argument("--noise", type=str, default="gaussian")
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument(
        "--h5_indices",
        type=str,
        default="",
        help="Optional comma list such as 110,111. Empty means all matched h5 files.",
    )
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_datasets_per_file", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--max_node_attention_batches",
        type=int,
        default=60000,
        help=(
            "Cap batch_size * (num_observations + 1) to avoid CUDA attention "
            "kernel launch limits. Set <=0 to disable."
        ),
    )
    parser.add_argument("--num_perm_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no_standardize", action="store_true")
    parser.add_argument("--output_dir", type=str, default="bak_qparam_perm_results")
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
    perm_counts_csv = output_dir / f"{prefix}_perm_counts.csv"
    global_counts_csv = output_dir / f"{prefix}_global_perm_counts.csv"

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
        "p_param_json",
        "q_param_raw_json",
        "q_param_log_alpha_json",
        "l_param_json",
        "edge_probs_json",
        "mean_prob_json",
        *[f"mean_prob_{i}to{j}" for i, j in EDGE_ORDER_3VAR],
        "num_perm_samples",
        "num_unique_perms",
        "model_path",
        "bak_model_file",
    ]
    perm_fieldnames = [
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
        "perm_rank_node_to_position_json",
        "perm_matrix_json",
        "perm_mask_json",
        "perm_prob_json",
        "root_to_leaf_json",
        "leaf_to_root_json",
        "root_node",
        "leaf_node",
        "count",
        "frequency",
        "num_perm_samples",
    ]

    global_counter: Counter[Tuple[str, Tuple[int, ...]]] = Counter()

    print("=" * 100)
    print("Inspecting bak Q_param and sampled permutations")
    print(f"model:          {model_path}")
    print(f"bak file:       {Path(args.bak_model_file).expanduser().resolve()}")
    print(f"device:         {device}")
    print(f"num_perm:       {num_perm_samples}")
    print(f"h5 files:       {len(h5_files)}")
    print(f"dataset csv:    {dataset_csv}")
    print(f"perm count csv: {perm_counts_csv}")
    print("=" * 100)

    with open(dataset_csv, "w", newline="", encoding="utf-8") as f_dataset, open(
        perm_counts_csv, "w", newline="", encoding="utf-8"
    ) as f_perm:
        dataset_writer = csv.DictWriter(f_dataset, fieldnames=dataset_fieldnames)
        perm_writer = csv.DictWriter(f_perm, fieldnames=perm_fieldnames)
        dataset_writer.writeheader()
        perm_writer.writeheader()

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
                        out = inspect_qparam_and_perm(
                            model=model,
                            inputs=inputs,
                            num_perm_samples=num_perm_samples,
                        )

                    perms = out["perm"]
                    ranks = perms.argmax(axis=-1)  # [num_perm_samples, batch, d]

                    for local_idx in range(end - start):
                        data_idx = start + local_idx
                        true_graph = labels[local_idx].astype(int)
                        np.fill_diagonal(true_graph, 0)
                        true_graph_json = json_dumps(true_graph.tolist())
                        true_edges = matrix_to_edges(true_graph)
                        true_flat = flat_graph_columns("true", true_graph)

                        rank_samples = [tuple(int(x) for x in ranks[s, local_idx].tolist()) for s in range(num_perm_samples)]
                        counter = Counter(rank_samples)

                        dataset_row = {
                            "h5_path": str(h5_path),
                            "h5_file": h5_path.name,
                            **h5_meta,
                            "data_idx": data_idx,
                            "true_graph_json": true_graph_json,
                            "true_edges": true_edges,
                            **true_flat,
                            "p_param_json": json_dumps(out["p_param"][local_idx].tolist()),
                            "q_param_raw_json": json_dumps(out["q_param_raw"][local_idx].tolist()),
                            "q_param_log_alpha_json": json_dumps(out["q_param_log_alpha"][local_idx].tolist()),
                            "l_param_json": json_dumps(out["l_param"][local_idx].tolist()),
                            "edge_probs_json": json_dumps(out["edge_probs"][local_idx].tolist()),
                            "mean_prob_json": json_dumps(out["mean_prob"][local_idx].tolist()),
                            **flat_matrix_columns("mean_prob", out["mean_prob"][local_idx]),
                            "num_perm_samples": num_perm_samples,
                            "num_unique_perms": len(counter),
                            "model_path": str(model_path),
                            "bak_model_file": str(Path(args.bak_model_file).expanduser().resolve()),
                        }
                        dataset_writer.writerow(dataset_row)

                        for rank, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
                            rank_arr = np.asarray(rank, dtype=int)
                            leaf_to_root = np.argsort(rank_arr).astype(int).tolist()
                            root_to_leaf = np.argsort(-rank_arr).astype(int).tolist()
                            global_counter[(true_graph_json, rank)] += int(count)
                            sample_idx = rank_samples.index(rank)
                            perm_writer.writerow(
                                {
                                    "h5_path": str(h5_path),
                                    "h5_file": h5_path.name,
                                    **h5_meta,
                                    "data_idx": data_idx,
                                    "true_graph_json": true_graph_json,
                                    "true_edges": true_edges,
                                    **true_flat,
                                    "perm_rank_node_to_position_json": json_dumps(list(rank)),
                                    "perm_matrix_json": json_dumps(rank_to_perm_matrix(rank)),
                                    "perm_mask_json": json_dumps(out["all_masks"][sample_idx, local_idx].astype(int).tolist()),
                                    "perm_prob_json": json_dumps(out["all_probs"][sample_idx, local_idx].tolist()),
                                    "root_to_leaf_json": json_dumps(root_to_leaf),
                                    "leaf_to_root_json": json_dumps(leaf_to_root),
                                    "root_node": int(root_to_leaf[0]),
                                    "leaf_node": int(root_to_leaf[-1]),
                                    "count": int(count),
                                    "frequency": float(count) / float(num_perm_samples),
                                    "num_perm_samples": num_perm_samples,
                                }
                            )

    with open(global_counts_csv, "w", newline="", encoding="utf-8") as f_global:
        fieldnames = [
            "true_graph_json",
            "true_edges",
            "perm_rank_node_to_position_json",
            "root_to_leaf_json",
            "leaf_to_root_json",
            "root_node",
            "leaf_node",
            "count",
        ]
        writer = csv.DictWriter(f_global, fieldnames=fieldnames)
        writer.writeheader()
        for (true_graph_json, rank), count in sorted(global_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            true_graph = np.asarray(json.loads(true_graph_json), dtype=int)
            rank_arr = np.asarray(rank, dtype=int)
            leaf_to_root = np.argsort(rank_arr).astype(int).tolist()
            root_to_leaf = np.argsort(-rank_arr).astype(int).tolist()
            writer.writerow(
                {
                    "true_graph_json": true_graph_json,
                    "true_edges": matrix_to_edges(true_graph),
                    "perm_rank_node_to_position_json": json_dumps(list(rank)),
                    "root_to_leaf_json": json_dumps(root_to_leaf),
                    "leaf_to_root_json": json_dumps(leaf_to_root),
                    "root_node": int(root_to_leaf[0]),
                    "leaf_node": int(root_to_leaf[-1]),
                    "count": int(count),
                }
            )

    print("=" * 100)
    print("Done")
    print(f"Wrote: {dataset_csv}")
    print(f"Wrote: {perm_counts_csv}")
    print(f"Wrote: {global_counts_csv}")


if __name__ == "__main__":
    main()
