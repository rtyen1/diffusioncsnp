#!/usr/bin/env python3
"""Generate mixed-distribution synthetic training data for causal discovery.

Each HDF5 shard has a fixed shape:

    data:     [num_datasets, num_samples, num_nodes]
    label:    [num_datasets, num_nodes, num_nodes]
    metadata: [num_datasets]  (one compact record per dataset)

Different shards under the same split directory may have different node counts.
Within every dataset, the graph, SEM, functions/weights, noise configuration,
and local-R2 distribution are sampled afresh.

All SEM families use the same 50/50 ER or scale-free graph sampler. Given a
sampled DAG, the GP branch keeps the repository's current
``GPFunctionGenerator.generate_data`` path. Linear/MLP datasets follow the
article-style construction:
local R2 is calibrated from the current dataset's empirical signal/noise
variances, and every node is standardised immediately after it is generated.
Consequently, rows are exchangeable but not strictly independent after these
dataset-level transformations.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import numpy as np
import tensorflow as tf
from tqdm import trange

from ml2_meta_causal_discovery.datasets.functions_generator import GPFunctionGenerator


SEM_GP = 0
SEM_MLP = 1
SEM_LINEAR = 2

GRAPH_ER = 0
GRAPH_SCALE_FREE = 1

NOISE_GP_CURRENT = -1
NOISE_HOMOGENEOUS = 0
NOISE_HETEROGENEOUS = 1

SEM_NAMES = {
    SEM_GP: "gp",
    SEM_MLP: "mlp",
    SEM_LINEAR: "linear",
}
GRAPH_NAMES = {
    GRAPH_ER: "er",
    GRAPH_SCALE_FREE: "scale_free",
}
NOISE_MODE_NAMES = {
    NOISE_GP_CURRENT: "gp_current",
    NOISE_HOMOGENEOUS: "homogeneous",
    NOISE_HETEROGENEOUS: "heterogeneous",
}

METADATA_DTYPE = np.dtype(
    [
        ("dataset_seed", "<u8"),
        ("sem_type", "u1"),
        ("graph_type", "u1"),
        ("noise_mode", "i1"),
        ("mean_target_r2", "<f4"),
    ]
)


@dataclass(frozen=True)
class NoiseSpec:
    family: str
    alpha: float = math.nan
    beta: float = math.nan


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def json_compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_probabilities(gp: float, mlp: float, linear: float) -> np.ndarray:
    probs = np.asarray([gp, mlp, linear], dtype=np.float64)
    if np.any(probs < 0):
        raise ValueError("SEM probabilities must be non-negative.")
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError(
            f"SEM probabilities must sum to 1, got {probs.tolist()} "
            f"(sum={probs.sum():.8f})."
        )
    return probs


def sample_signed_weights(
    rng: np.random.Generator,
    shape: Sequence[int],
    amplitude: float,
) -> np.ndarray:
    magnitude = rng.uniform(1.0 - amplitude, 1.0 + amplitude, size=shape)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=shape)
    return magnitude * signs


def sample_graph(
    rng: np.random.Generator,
    num_nodes: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Sample an ER or static power-law skeleton and orient it by a random order."""
    pairs = np.asarray(
        [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)],
        dtype=np.int64,
    )
    max_edges = int(pairs.shape[0])
    num_edges = int(rng.integers(0, min(4 * num_nodes, max_edges) + 1))
    graph_type = GRAPH_ER if rng.random() < 0.5 else GRAPH_SCALE_FREE

    skeleton = np.zeros((num_nodes, num_nodes), dtype=np.int8)
    if num_edges > 0:
        if graph_type == GRAPH_ER or num_edges == max_edges:
            selected = rng.choice(max_edges, size=num_edges, replace=False)
        else:
            alpha = float(rng.uniform(2.0, 3.0))
            node_weights = rng.pareto(alpha, size=num_nodes) + 1.0
            pair_weights = node_weights[pairs[:, 0]] * node_weights[pairs[:, 1]]
            pair_probs = pair_weights / pair_weights.sum()
            selected = rng.choice(
                max_edges,
                size=num_edges,
                replace=False,
                p=pair_probs,
            )
        selected_pairs = pairs[np.asarray(selected, dtype=np.int64)]
        skeleton[selected_pairs[:, 0], selected_pairs[:, 1]] = 1
        skeleton[selected_pairs[:, 1], selected_pairs[:, 0]] = 1

    topo_order = rng.permutation(num_nodes).astype(np.int64)
    position = np.empty(num_nodes, dtype=np.int64)
    position[topo_order] = np.arange(num_nodes)
    dag = np.zeros_like(skeleton)
    for i, j in pairs:
        if skeleton[i, j] == 0:
            continue
        if position[i] < position[j]:
            dag[i, j] = 1
        else:
            dag[j, i] = 1
    return dag, topo_order, graph_type


def sample_noise_spec(rng: np.random.Generator) -> NoiseSpec:
    family = str(rng.choice(np.asarray(["gaussian", "uniform", "beta"])))
    if family == "beta":
        return NoiseSpec(
            family=family,
            alpha=float(rng.uniform(1.0, 10.0)),
            beta=float(rng.uniform(1.0, 10.0)),
        )
    return NoiseSpec(family=family)


def sample_unit_noise(
    rng: np.random.Generator,
    spec: NoiseSpec,
    size: int,
) -> np.ndarray:
    """Draw IID noise with population mean zero and population variance one."""
    if spec.family == "gaussian":
        return rng.standard_normal(size)
    if spec.family == "uniform":
        return rng.uniform(-1.0, 1.0, size=size) * math.sqrt(3.0)
    if spec.family == "beta":
        raw = rng.beta(spec.alpha, spec.beta, size=size)
        mean = spec.alpha / (spec.alpha + spec.beta)
        variance = (
            spec.alpha
            * spec.beta
            / (
                (spec.alpha + spec.beta) ** 2
                * (spec.alpha + spec.beta + 1.0)
            )
        )
        return (raw - mean) / math.sqrt(variance)
    raise ValueError(f"Unknown noise family: {spec.family}")


def apply_activation(x: np.ndarray, activation: str) -> np.ndarray:
    if activation == "tanh":
        return np.tanh(x)
    if activation == "hardtanh":
        return np.clip(x, -1.0, 1.0)
    if activation == "sigmoid":
        clipped = np.clip(x, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clipped))
    if activation == "hardsigmoid":
        return np.clip(x / 6.0 + 0.5, 0.0, 1.0)
    raise ValueError(f"Unknown activation: {activation}")


def standardise_empirically(values: np.ndarray) -> np.ndarray:
    """Standardise one generated node using the current dataset."""
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(std) or std < 1e-8:
        raise ValueError(f"Degenerate empirical standard deviation: {std}")
    return (values - mean) / std


def noise_scale_for_exact_empirical_r2(
    signal: np.ndarray,
    noise: np.ndarray,
    target_r2: float,
) -> float:
    """Choose a positive scale giving the requested empirical local R2.

    The covariance term is retained, so the equality holds for the current
    finite dataset rather than only in expectation.
    """
    signal_centered = signal - signal.mean()
    noise_centered = noise - noise.mean()
    signal_variance = float(np.mean(signal_centered**2))
    noise_variance = float(np.mean(noise_centered**2))
    covariance = float(np.mean(signal_centered * noise_centered))
    if signal_variance < 1e-8 or noise_variance < 1e-8:
        raise ValueError(
            "Cannot calibrate local R2 with degenerate signal or noise variance."
        )
    target_extra_variance = signal_variance * (1.0 - target_r2) / target_r2
    discriminant = covariance**2 + noise_variance * target_extra_variance
    return (-covariance + math.sqrt(max(discriminant, 0.0))) / noise_variance


def generate_parametric_sem(
    *,
    dag_topological: np.ndarray,
    num_samples: int,
    sem_type: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int, float]:
    """Generate an article-style linear or one-hidden-layer MLP SEM."""
    num_nodes = int(dag_topological.shape[0])
    observed = np.zeros((num_samples, num_nodes), dtype=np.float64)

    weight_amplitude = float(rng.uniform(0.0, 0.9))
    r2_alpha = float(rng.uniform(1.0, 10.0))
    r2_beta = float(rng.uniform(1.0, 10.0))
    noise_mode = (
        NOISE_HOMOGENEOUS if rng.random() < 0.5 else NOISE_HETEROGENEOUS
    )
    shared_noise_spec = (
        sample_noise_spec(rng) if noise_mode == NOISE_HOMOGENEOUS else None
    )

    hidden_dim = int(rng.integers(1, 65)) if sem_type == SEM_MLP else 0
    activation = (
        str(
            rng.choice(
                np.asarray(["tanh", "hardtanh", "sigmoid", "hardsigmoid"])
            )
        )
        if sem_type == SEM_MLP
        else ""
    )

    target_r2_values: List[float] = []
    for node in range(num_nodes):
        parent_idx = np.flatnonzero(dag_topological[:, node])
        noise_spec = (
            shared_noise_spec
            if shared_noise_spec is not None
            else sample_noise_spec(rng)
        )
        assert noise_spec is not None

        observed_noise = sample_unit_noise(rng, noise_spec, num_samples)

        if parent_idx.size == 0:
            observed[:, node] = standardise_empirically(observed_noise)
            continue

        observed_parents = observed[:, parent_idx]

        signal_observed: np.ndarray
        for _ in range(20):
            if sem_type == SEM_LINEAR:
                weights = sample_signed_weights(
                    rng,
                    (parent_idx.size,),
                    weight_amplitude,
                )
                signal_observed = observed_parents @ weights
            elif sem_type == SEM_MLP:
                weights_in = sample_signed_weights(
                    rng,
                    (parent_idx.size, hidden_dim),
                    weight_amplitude,
                )
                weights_out = sample_signed_weights(
                    rng,
                    (hidden_dim,),
                    weight_amplitude,
                )
                signal_observed = (
                    apply_activation(observed_parents @ weights_in, activation)
                    @ weights_out
                )
            else:
                raise ValueError(f"Unsupported parametric SEM type: {sem_type}")

            signal_variance = float(np.var(signal_observed))
            if np.isfinite(signal_variance) and signal_variance >= 1e-8:
                break
        else:
            raise RuntimeError(
                f"Could not sample a non-degenerate signal for node {node}."
            )

        target_r2 = 0.1 + 0.8 * float(rng.beta(r2_alpha, r2_beta))
        target_r2_values.append(target_r2)
        noise_scale = noise_scale_for_exact_empirical_r2(
            signal_observed,
            observed_noise,
            target_r2,
        )
        observed_values = signal_observed + noise_scale * observed_noise
        observed[:, node] = standardise_empirically(observed_values)

    mean_target_r2 = (
        float(np.mean(target_r2_values)) if target_r2_values else math.nan
    )
    return observed.astype(np.float32), noise_mode, mean_target_r2


def generate_gp_sem_current(
    *,
    dag_topological: np.ndarray,
    num_samples: int,
    dataset_seed: int | None,
) -> np.ndarray:
    """Call the repository's current GP generation path unchanged."""
    if dataset_seed is not None:
        seed32 = int(dataset_seed % np.iinfo(np.uint32).max)
        np.random.seed(seed32)
        tf.random.set_seed(seed32)

    generator = GPFunctionGenerator(
        num_variables=int(dag_topological.shape[0]),
        num_samples=num_samples,
        interventions=False,
    )
    data = generator.generate_data(
        causal_graph=dag_topological,
        num_int_samples=num_samples,
    )
    return np.asarray(data, dtype=np.float32)


def is_dag(adjacency: np.ndarray) -> bool:
    indegree = adjacency.sum(axis=0).astype(np.int64)
    queue = list(np.flatnonzero(indegree == 0))
    visited = 0
    while queue:
        node = int(queue.pop())
        visited += 1
        for child in np.flatnonzero(adjacency[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(int(child))
    return visited == adjacency.shape[0]


def validate_shard_arrays(
    *,
    data: np.ndarray,
    labels: np.ndarray,
    metadata: np.ndarray,
    num_datasets: int,
    num_samples: int,
    num_nodes: int,
) -> None:
    """Fail before writing if a generated shard violates its contract."""
    expected_data_shape = (num_datasets, num_samples, num_nodes)
    expected_label_shape = (num_datasets, num_nodes, num_nodes)
    if data.shape != expected_data_shape:
        raise RuntimeError(
            f"Data shape mismatch: expected {expected_data_shape}, got {data.shape}."
        )
    if labels.shape != expected_label_shape:
        raise RuntimeError(
            f"Label shape mismatch: expected {expected_label_shape}, got {labels.shape}."
        )
    if metadata.shape != (num_datasets,):
        raise RuntimeError(
            f"Metadata shape mismatch: expected {(num_datasets,)}, got {metadata.shape}."
        )
    if not np.isfinite(data).all():
        raise RuntimeError("Generated shard contains NaN or Inf.")
    if not np.isin(labels, [0, 1]).all():
        raise RuntimeError("Graph labels must be binary.")
    if np.any(np.diagonal(labels, axis1=1, axis2=2) != 0):
        raise RuntimeError("Graph labels must have a zero diagonal.")
    if not all(is_dag(graph) for graph in labels):
        raise RuntimeError("Generated shard contains a cyclic graph.")
    if np.unique(metadata["dataset_seed"]).size != num_datasets:
        raise RuntimeError("Dataset seeds must be unique within each HDF5 shard.")
    if not np.isin(metadata["sem_type"], list(SEM_NAMES)).all():
        raise RuntimeError("Generated shard contains an unknown SEM code.")
    if not np.isin(metadata["graph_type"], list(GRAPH_NAMES)).all():
        raise RuntimeError("Generated shard contains an unknown graph-type code.")
    if not np.isin(metadata["noise_mode"], list(NOISE_MODE_NAMES)).all():
        raise RuntimeError("Generated shard contains an unknown noise-mode code.")

    gp_mask = metadata["sem_type"] == SEM_GP
    parametric_mask = ~gp_mask
    if np.any(metadata["noise_mode"][gp_mask] != NOISE_GP_CURRENT):
        raise RuntimeError("Every GP dataset must use the current GP noise mode.")
    if np.any(np.isfinite(metadata["mean_target_r2"][gp_mask])):
        raise RuntimeError("GP datasets must not report a controlled local R2.")
    if np.any(
        ~np.isin(
            metadata["noise_mode"][parametric_mask],
            [NOISE_HOMOGENEOUS, NOISE_HETEROGENEOUS],
        )
    ):
        raise RuntimeError("Linear/MLP datasets have an invalid noise mode.")

    if np.any(parametric_mask):
        has_nonroot = labels.sum(axis=(1, 2)) > 0
        controlled_r2_mask = parametric_mask & has_nonroot
        controlled_r2 = metadata["mean_target_r2"][controlled_r2_mask]
        if np.any(~np.isfinite(controlled_r2)):
            raise RuntimeError(
                "Every non-empty Linear/MLP graph must report a mean target R2."
            )
        if np.any((controlled_r2 < 0.1) | (controlled_r2 > 0.9)):
            raise RuntimeError("Mean target R2 must lie in [0.1, 0.9].")

        parametric_data = data[parametric_mask].astype(np.float64)
        max_abs_mean = float(np.max(np.abs(parametric_data.mean(axis=1))))
        max_std_error = float(
            np.max(np.abs(parametric_data.std(axis=1) - 1.0))
        )
        if max_abs_mean > 1e-5 or max_std_error > 1e-5:
            raise RuntimeError(
                "Linear/MLP nodes were not standardised correctly: "
                f"max_abs_mean={max_abs_mean}, max_std_error={max_std_error}."
            )


def generate_dataset(
    *,
    num_nodes: int,
    num_samples: int,
    sem_probs: np.ndarray,
    dataset_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.void]:
    rng = np.random.default_rng(dataset_seed)
    sem_type = int(rng.choice([SEM_GP, SEM_MLP, SEM_LINEAR], p=sem_probs))
    dag, topo_order, graph_type = sample_graph(rng, num_nodes)
    if not is_dag(dag):
        raise RuntimeError("Internal error: sampled graph is not a DAG.")
    dag_topological = dag[np.ix_(topo_order, topo_order)]
    if np.any(np.tril(dag_topological) != 0):
        raise RuntimeError(
            "Internal error: topologically reordered DAG is not upper triangular."
        )

    if sem_type == SEM_GP:
        data_topological = generate_gp_sem_current(
            dag_topological=dag_topological,
            num_samples=num_samples,
            dataset_seed=dataset_seed,
        )
        noise_mode = NOISE_GP_CURRENT
        mean_target_r2 = math.nan
    else:
        data_topological, noise_mode, mean_target_r2 = generate_parametric_sem(
            dag_topological=dag_topological,
            num_samples=num_samples,
            sem_type=sem_type,
            rng=rng,
        )
    data = np.empty_like(data_topological)
    data[:, topo_order] = data_topological
    if not np.isfinite(data).all():
        raise RuntimeError("Generated data contains NaN or Inf.")

    metadata = np.zeros((), dtype=METADATA_DTYPE)
    metadata["dataset_seed"] = np.uint64(dataset_seed)
    metadata["sem_type"] = np.uint8(sem_type)
    metadata["graph_type"] = np.uint8(graph_type)
    metadata["noise_mode"] = np.int8(noise_mode)
    metadata["mean_target_r2"] = np.float32(mean_target_r2)
    return data, dag.astype(np.int8), metadata


def compression_kwargs(name: str) -> Dict[str, object]:
    if name == "none":
        return {}
    if name == "lzf":
        return {"compression": "lzf"}
    if name == "gzip":
        return {"compression": "gzip", "compression_opts": 1}
    raise ValueError(f"Unknown compression: {name}")


def write_shard(
    *,
    output_path: Path,
    num_nodes: int,
    num_samples: int,
    num_datasets: int,
    sem_probs: np.ndarray,
    shard_seed: int,
    compression: str,
    overwrite: bool,
    show_progress: bool = True,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"[SKIP] {output_path}")
        return

    shard_rng = np.random.default_rng(shard_seed)
    dataset_seeds = shard_rng.integers(
        0,
        np.iinfo(np.uint64).max,
        size=num_datasets,
        dtype=np.uint64,
    )
    data = np.empty(
        (num_datasets, num_samples, num_nodes),
        dtype=np.float32,
    )
    labels = np.empty(
        (num_datasets, num_nodes, num_nodes),
        dtype=np.int8,
    )
    metadata = np.empty(num_datasets, dtype=METADATA_DTYPE)

    description = f"d={num_nodes} {output_path.name}"
    for index in trange(
        num_datasets,
        desc=description,
        disable=not show_progress,
    ):
        sample, dag, meta = generate_dataset(
            num_nodes=num_nodes,
            num_samples=num_samples,
            sem_probs=sem_probs,
            dataset_seed=int(dataset_seeds[index]),
        )
        data[index] = sample
        labels[index] = dag
        metadata[index] = meta

    validate_shard_arrays(
        data=data,
        labels=labels,
        metadata=metadata,
        num_datasets=num_datasets,
        num_samples=num_samples,
        num_nodes=num_nodes,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = compression_kwargs(compression)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("data", data=data, **kwargs)
        handle.create_dataset("label", data=labels, **kwargs)
        handle.create_dataset("metadata", data=metadata)
        handle.attrs["distribution"] = "diverse_causal_pretrain_v1"
        handle.attrs["n_nodes"] = int(num_nodes)
        handle.attrs["num_samples"] = int(num_samples)
        handle.attrs["num_datasets"] = int(num_datasets)
        handle.attrs["shard_seed"] = int(shard_seed)
        handle.attrs["label_convention"] = "label[i, j] = 1 means i -> j"
        handle.attrs["sem_type_codes"] = json_compact(SEM_NAMES)
        handle.attrs["graph_type_codes"] = json_compact(GRAPH_NAMES)
        handle.attrs["noise_mode_codes"] = json_compact(NOISE_MODE_NAMES)
        handle.attrs["sem_probabilities"] = json_compact(
            {
                "gp": float(sem_probs[SEM_GP]),
                "mlp": float(sem_probs[SEM_MLP]),
                "linear": float(sem_probs[SEM_LINEAR]),
            }
        )

    sem_counts = {
        SEM_NAMES[sem]: int(np.sum(metadata["sem_type"] == sem))
        for sem in SEM_NAMES
    }
    print(
        f"[WRITE] {output_path} data={data.shape} label={labels.shape} "
        f"sem_counts={sem_counts}"
    )


def write_manifest(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    node_counts: Sequence[int],
    sem_probs: np.ndarray,
    overwrite: bool,
) -> None:
    manifest_path = output_dir / "generation_config.json"
    if manifest_path.exists() and not overwrite:
        return
    config = {
        "distribution": "diverse_causal_pretrain_v1",
        "split": args.split,
        "node_counts": list(node_counts),
        "num_samples_stored": args.num_samples,
        "training_sample_size_range": [100, args.num_samples],
        "num_datasets_per_node_count": args.num_datasets_per_node_count,
        "datasets_per_file": args.datasets_per_file,
        "seed": args.seed,
        "sem_probabilities": {
            "gp": float(sem_probs[SEM_GP]),
            "mlp": float(sem_probs[SEM_MLP]),
            "linear": float(sem_probs[SEM_LINEAR]),
        },
        "graph_probabilities_all_sem_families": {
            "er": 0.5,
            "scale_free": 0.5,
        },
        "edge_count": "UniformInteger(0, min(4*d, d*(d-1)/2))",
        "sample_statement": (
            "Rows share one sampled SCM and independent raw exogenous-noise draws. "
            "Article-style empirical R2 calibration and node standardisation make "
            "the saved rows exchangeable rather than strictly independent."
        ),
        "hdf5_datasets": {
            "data": ["datasets_in_shard", args.num_samples, "num_nodes"],
            "label": ["datasets_in_shard", "num_nodes", "num_nodes"],
            "metadata": ["datasets_in_shard"],
        },
        "metadata_fields": list(METADATA_DTYPE.names or ()),
        "compression": args.compression,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed-shape HDF5 shards containing mixed GP/MLP/linear "
            "article-style causal datasets."
        )
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=(
            "ml2_meta_causal_discovery/datasets/data/synth_training_data/"
            "diverse_causal_pretrain_v1"
        ),
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--node_counts", type=str, default="3,4,5")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--num_datasets_per_node_count", type=int, required=True)
    parser.add_argument("--datasets_per_file", type=int, default=500)
    parser.add_argument("--gp_probability", type=float, default=0.35)
    parser.add_argument("--mlp_probability", type=float, default=0.50)
    parser.add_argument("--linear_probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compression",
        choices=["none", "lzf", "gzip"],
        default="none",
        help="'none' is fastest; 'lzf' is a fast compression option.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Generate independent HDF5 shards in parallel processes. "
            "Start with 1-2 because exact GP sampling is also multithreaded."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    node_counts = parse_int_list(args.node_counts)
    if not node_counts or any(node <= 0 for node in node_counts):
        raise ValueError("--node_counts must contain positive integers.")
    if args.num_samples <= 1:
        raise ValueError("--num_samples must be greater than 1.")
    if args.num_datasets_per_node_count <= 0:
        raise ValueError("--num_datasets_per_node_count must be positive.")
    if args.datasets_per_file <= 0:
        raise ValueError("--datasets_per_file must be positive.")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")

    sem_probs = validate_probabilities(
        args.gp_probability,
        args.mlp_probability,
        args.linear_probability,
    )
    output_dir = Path(args.save_root).expanduser().resolve() / args.split
    write_manifest(
        output_dir=output_dir,
        args=args,
        node_counts=node_counts,
        sem_probs=sem_probs,
        overwrite=args.overwrite,
    )

    master_rng = np.random.default_rng(args.seed)
    print("=" * 100)
    print("Generating diverse causal pretraining data")
    print(f"output_dir:                  {output_dir}")
    print(f"node_counts:                 {node_counts}")
    print(f"num_samples per dataset:     {args.num_samples}")
    print(f"datasets per node count:     {args.num_datasets_per_node_count}")
    print(f"datasets per HDF5:           {args.datasets_per_file}")
    print(f"parallel shard workers:      {args.workers}")
    print(
        "SEM probabilities:          "
        f"gp={sem_probs[SEM_GP]:.3f}, "
        f"mlp={sem_probs[SEM_MLP]:.3f}, "
        f"linear={sem_probs[SEM_LINEAR]:.3f}"
    )
    print(f"compression:                 {args.compression}")
    print("=" * 100)

    shard_jobs: List[Dict[str, Any]] = []
    for num_nodes in node_counts:
        num_parts = math.ceil(
            args.num_datasets_per_node_count / args.datasets_per_file
        )
        for part_index in range(num_parts):
            start = part_index * args.datasets_per_file
            end = min(
                args.num_datasets_per_node_count,
                start + args.datasets_per_file,
            )
            shard_num_datasets = end - start
            shard_seed = int(
                master_rng.integers(0, np.iinfo(np.uint32).max)
            )
            output_path = (
                output_dir
                / f"d{num_nodes}_shard_{part_index:05d}.hdf5"
            )
            shard_jobs.append(
                {
                    "output_path": output_path,
                    "num_nodes": num_nodes,
                    "num_samples": args.num_samples,
                    "num_datasets": shard_num_datasets,
                    "sem_probs": sem_probs,
                    "shard_seed": shard_seed,
                    "compression": args.compression,
                    "overwrite": args.overwrite,
                    "show_progress": args.workers == 1,
                }
            )

    if args.workers == 1:
        for job in shard_jobs:
            write_shard(**job)
        return

    # "spawn" avoids forking an already-initialised TensorFlow runtime.
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
    ) as executor:
        futures = [executor.submit(write_shard, **job) for job in shard_jobs]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
