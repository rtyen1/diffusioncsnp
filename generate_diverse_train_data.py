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
``GPFunctionGenerator.generate_data`` path. Linear, additive MLP, and random
Fourier feature (RFF) datasets follow the article-style construction: local R2
is calibrated from the current dataset's empirical signal/noise variances.
Non-additive MLP datasets instead concatenate scaled exogenous noise to the
parents before applying the MLP. Every generated node, including GP outputs,
is empirically standardised before it is written.
Consequently, rows are exchangeable but not strictly independent after these
dataset-level transformations.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange

from ml2_meta_causal_discovery.datasets.functions_generator import GPFunctionGenerator


SEM_GP = 0
SEM_MLP = 1
SEM_LINEAR = 2
SEM_MLP_NONADDITIVE = 3
SEM_RFF = 4

GRAPH_ER = 0
GRAPH_SCALE_FREE = 1

NOISE_GP_CURRENT = -1
NOISE_HOMOGENEOUS = 0
NOISE_HETEROGENEOUS = 1

SEM_NAMES = {
    SEM_GP: "gp",
    SEM_MLP: "mlp",
    SEM_LINEAR: "linear",
    SEM_MLP_NONADDITIVE: "mlp_nonadditive",
    SEM_RFF: "rff",
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

MLP_DEPTHS = (1, 2)
MLP_HIDDEN_WIDTHS = (8, 16, 32, 64)
MLP_ACTIVATIONS = ("tanh", "hardtanh", "sigmoid", "hardsigmoid", "leaky_relu")
ACTIVATION_NAMES = {index: name for index, name in enumerate(MLP_ACTIVATIONS)}
ACTIVATION_CODES = {name: index for index, name in ACTIVATION_NAMES.items()}
RFF_FEATURE_COUNTS = (16, 32, 64)
NONADDITIVE_NOISE_SCALE_RANGE = (0.1, 2.0)
RFF_LENGTHSCALE_RANGE = (0.5, 2.0)

METADATA_DTYPE = np.dtype(
    [
        ("dataset_seed", "<u8"),
        ("sem_type", "u1"),
        ("graph_type", "u1"),
        ("noise_mode", "i1"),
        ("mean_target_r2", "<f4"),
        ("mlp_depth", "u1"),
        ("mlp_hidden_width", "u1"),
        ("activation", "i1"),
        ("mean_nonadditive_noise_scale", "<f4"),
        ("rff_num_features", "u1"),
        ("rff_lengthscale", "<f4"),
    ]
)


@dataclass(frozen=True)
class NoiseSpec:
    family: str
    alpha: float = math.nan
    beta: float = math.nan


@dataclass(frozen=True)
class SemDetails:
    noise_mode: int
    mean_target_r2: float = math.nan
    mlp_depth: int = 0
    mlp_hidden_width: int = 0
    activation: int = -1
    mean_nonadditive_noise_scale: float = math.nan
    rff_num_features: int = 0
    rff_lengthscale: float = math.nan


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def normalise_probability_vector(
    values: Sequence[float],
    *,
    expected_size: int,
    name: str,
) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.shape != (expected_size,):
        raise ValueError(
            f"{name} must contain {expected_size} values, "
            f"got {len(probabilities)}."
        )
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError(f"{name} must be finite and non-negative.")
    probability_sum = float(probabilities.sum())
    if not np.isclose(probability_sum, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"{name} must sum to 1, got {probabilities.tolist()} "
            f"(sum={probability_sum:.12f})."
        )
    return probabilities / probability_sum


def allocate_node_count_datasets(
    *,
    node_counts: Sequence[int],
    num_datasets_total: int,
    probabilities: Sequence[float],
) -> Dict[int, int]:
    """Allocate an exact integer total using stable largest remainders."""
    probs = normalise_probability_vector(
        probabilities,
        expected_size=len(node_counts),
        name="--node_count_probabilities",
    )

    expected = probs * num_datasets_total
    allocated = np.floor(expected).astype(np.int64)
    remainder = int(num_datasets_total - allocated.sum())
    remainder_order = np.argsort(-(expected - allocated), kind="stable")
    allocated[remainder_order[:remainder]] += 1

    missing_positive = (probs > 0.0) & (allocated == 0)
    if np.any(missing_positive):
        missing = [
            int(node_counts[index]) for index in np.flatnonzero(missing_positive)
        ]
        raise ValueError(
            "The requested total is too small to allocate at least one dataset "
            f"to positive-probability node counts {missing}."
        )
    if int(allocated.sum()) != num_datasets_total:
        raise RuntimeError("Internal error: node-count allocation changed the total.")
    return {
        int(node_count): int(count)
        for node_count, count in zip(node_counts, allocated)
    }


def resolve_node_dataset_counts(
    args: argparse.Namespace,
    node_counts: Sequence[int],
) -> Tuple[Dict[int, int], List[float]]:
    per_node_count = args.num_datasets_per_node_count
    total = args.num_datasets_total
    probability_text = args.node_count_probabilities

    if per_node_count is not None:
        if total is not None or probability_text is not None:
            raise ValueError(
                "Use either --num_datasets_per_node_count, or "
                "--num_datasets_total with optional --node_count_probabilities."
            )
        if per_node_count <= 0:
            raise ValueError("--num_datasets_per_node_count must be positive.")
        counts = {int(node_count): int(per_node_count) for node_count in node_counts}
        probabilities = [1.0 / len(node_counts)] * len(node_counts)
        return counts, probabilities

    if total is None:
        raise ValueError(
            "Pass --num_datasets_per_node_count, or pass --num_datasets_total "
            "with optional --node_count_probabilities."
        )
    if total <= 0:
        raise ValueError("--num_datasets_total must be positive.")
    probabilities = (
        parse_float_list(probability_text)
        if probability_text is not None
        else [1.0 / len(node_counts)] * len(node_counts)
    )
    normalised_probabilities = normalise_probability_vector(
        probabilities,
        expected_size=len(node_counts),
        name="--node_count_probabilities",
    )
    counts = allocate_node_count_datasets(
        node_counts=node_counts,
        num_datasets_total=total,
        probabilities=normalised_probabilities,
    )
    return counts, normalised_probabilities.tolist()


def json_compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_probabilities(
    gp: float,
    mlp: float,
    linear: float,
    mlp_nonadditive: float = 0.0,
    rff: float = 0.0,
) -> np.ndarray:
    return normalise_probability_vector(
        [gp, mlp, linear, mlp_nonadditive, rff],
        expected_size=len(SEM_NAMES),
        name="SEM probabilities",
    )


def sem_probability_dict(sem_probs: np.ndarray) -> Dict[str, float]:
    return {
        SEM_NAMES[sem_type]: float(sem_probs[sem_type])
        for sem_type in SEM_NAMES
    }


def distribution_name(sem_probs: np.ndarray) -> str:
    del sem_probs  # Kept as an argument so callers retain the existing API.
    return "diverse_causal_pretrain_v2"


def finite_summary(values: np.ndarray) -> Dict[str, float | None]:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(finite_values)),
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
    }


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
    if activation == "leaky_relu":
        return np.where(x >= 0.0, x, 0.2 * x)
    raise ValueError(f"Unknown activation: {activation}")


def sample_mlp_architecture(
    rng: np.random.Generator,
) -> Tuple[int, int, str]:
    return (
        int(rng.choice(np.asarray(MLP_DEPTHS))),
        int(rng.choice(np.asarray(MLP_HIDDEN_WIDTHS))),
        str(rng.choice(np.asarray(MLP_ACTIVATIONS))),
    )


def random_mlp_forward(
    inputs: np.ndarray,
    *,
    depth: int,
    hidden_width: int,
    activation: str,
    weight_amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evaluate one independently initialised random MLP for one causal node."""
    hidden = np.asarray(inputs, dtype=np.float64)
    for _ in range(depth):
        weights = sample_signed_weights(
            rng,
            (hidden.shape[1], hidden_width),
            weight_amplitude,
        )
        hidden = apply_activation(hidden @ weights, activation)
    weights_out = sample_signed_weights(
        rng,
        (hidden_width,),
        weight_amplitude,
    )
    return hidden @ weights_out


def sample_log_uniform(
    rng: np.random.Generator,
    lower: float,
    upper: float,
) -> float:
    return float(np.exp(rng.uniform(np.log(lower), np.log(upper))))


def random_rff_forward(
    inputs: np.ndarray,
    *,
    num_features: int,
    lengthscale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evaluate one independently initialised random Fourier feature function."""
    frequencies = rng.normal(
        loc=0.0,
        scale=1.0 / lengthscale,
        size=(inputs.shape[1], num_features),
    )
    phases = rng.uniform(0.0, 2.0 * math.pi, size=(num_features,))
    output_weights = rng.standard_normal(num_features)
    features = math.sqrt(2.0 / num_features) * np.cos(
        inputs @ frequencies + phases
    )
    return features @ output_weights


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
) -> Tuple[np.ndarray, SemDetails]:
    """Generate one linear, MLP, non-additive MLP, or RFF SEM."""
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

    mlp_depth = 0
    mlp_hidden_width = 0
    activation = ""
    if sem_type in (SEM_MLP, SEM_MLP_NONADDITIVE):
        mlp_depth, mlp_hidden_width, activation = sample_mlp_architecture(rng)

    rff_num_features = 0
    rff_lengthscale = math.nan
    if sem_type == SEM_RFF:
        rff_num_features = int(rng.choice(np.asarray(RFF_FEATURE_COUNTS)))
        rff_lengthscale = sample_log_uniform(
            rng,
            RFF_LENGTHSCALE_RANGE[0],
            RFF_LENGTHSCALE_RANGE[1],
        )

    target_r2_values: List[float] = []
    nonadditive_noise_scales: List[float] = []
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

        if sem_type == SEM_MLP_NONADDITIVE:
            noise_scale = float(
                rng.uniform(
                    NONADDITIVE_NOISE_SCALE_RANGE[0],
                    NONADDITIVE_NOISE_SCALE_RANGE[1],
                )
            )
            mlp_inputs = np.concatenate(
                [observed_parents, noise_scale * observed_noise[:, None]],
                axis=1,
            )
            for _ in range(20):
                observed_values = random_mlp_forward(
                    mlp_inputs,
                    depth=mlp_depth,
                    hidden_width=mlp_hidden_width,
                    activation=activation,
                    weight_amplitude=weight_amplitude,
                    rng=rng,
                )
                output_variance = float(np.var(observed_values))
                if np.isfinite(output_variance) and output_variance >= 1e-8:
                    break
            else:
                raise RuntimeError(
                    "Could not sample a non-degenerate non-additive MLP "
                    f"output for node {node}."
                )
            observed[:, node] = standardise_empirically(observed_values)
            nonadditive_noise_scales.append(noise_scale)
            continue

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
                signal_observed = random_mlp_forward(
                    observed_parents,
                    depth=mlp_depth,
                    hidden_width=mlp_hidden_width,
                    activation=activation,
                    weight_amplitude=weight_amplitude,
                    rng=rng,
                )
            elif sem_type == SEM_RFF:
                signal_observed = random_rff_forward(
                    observed_parents,
                    num_features=rff_num_features,
                    lengthscale=rff_lengthscale,
                    rng=rng,
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
    mean_nonadditive_noise_scale = (
        float(np.mean(nonadditive_noise_scales))
        if nonadditive_noise_scales
        else math.nan
    )
    details = SemDetails(
        noise_mode=noise_mode,
        mean_target_r2=mean_target_r2,
        mlp_depth=mlp_depth,
        mlp_hidden_width=mlp_hidden_width,
        activation=ACTIVATION_CODES.get(activation, -1),
        mean_nonadditive_noise_scale=mean_nonadditive_noise_scale,
        rff_num_features=rff_num_features,
        rff_lengthscale=rff_lengthscale,
    )
    return observed.astype(np.float32), details


def generate_gp_sem_current(
    *,
    dag_topological: np.ndarray,
    num_samples: int,
    dataset_seed: int | None,
) -> np.ndarray:
    """Run the current GP mechanism, then apply the common output scaling."""
    if dataset_seed is not None:
        seed32 = int(dataset_seed % np.iinfo(np.uint32).max)
        np.random.seed(seed32)
        tf.random.set_seed(seed32)

    generator = GPFunctionGenerator(
        num_variables=int(dag_topological.shape[0]),
        num_samples=num_samples,
        interventions=False,
    )
    data = np.asarray(
        generator.generate_data(
            causal_graph=dag_topological,
            num_int_samples=num_samples,
        ),
        dtype=np.float64,
    )
    standardised = np.empty_like(data)
    for node in range(data.shape[1]):
        standardised[:, node] = standardise_empirically(data[:, node])
    return standardised.astype(np.float32)


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

    sem_types = metadata["sem_type"]
    gp_mask = sem_types == SEM_GP
    mlp_mask = np.isin(sem_types, [SEM_MLP, SEM_MLP_NONADDITIVE])
    nonadditive_mask = sem_types == SEM_MLP_NONADDITIVE
    rff_mask = sem_types == SEM_RFF
    additive_mask = np.isin(sem_types, [SEM_MLP, SEM_LINEAR, SEM_RFF])
    parametric_mask = ~gp_mask
    has_nonroot = labels.sum(axis=(1, 2)) > 0

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
        raise RuntimeError("A parametric SEM has an invalid noise mode.")

    if np.any(parametric_mask):
        controlled_r2_mask = additive_mask & has_nonroot
        controlled_r2 = metadata["mean_target_r2"][controlled_r2_mask]
        if np.any(~np.isfinite(controlled_r2)):
            raise RuntimeError(
                "Every non-empty additive graph must report a mean target R2."
            )
        if np.any((controlled_r2 < 0.1) | (controlled_r2 > 0.9)):
            raise RuntimeError("Mean target R2 must lie in [0.1, 0.9].")
        no_controlled_r2 = ~controlled_r2_mask
        if np.any(np.isfinite(metadata["mean_target_r2"][no_controlled_r2])):
            raise RuntimeError(
                "Only non-empty additive SEMs may report a controlled local R2."
            )

    standardised_data = data.astype(np.float64)
    max_abs_mean = float(np.max(np.abs(standardised_data.mean(axis=1))))
    max_std_error = float(
        np.max(np.abs(standardised_data.std(axis=1) - 1.0))
    )
    if max_abs_mean > 1e-5 or max_std_error > 1e-5:
        raise RuntimeError(
            "Generated nodes were not standardised correctly: "
            f"max_abs_mean={max_abs_mean}, max_std_error={max_std_error}."
        )

    if np.any(~np.isin(metadata["mlp_depth"][mlp_mask], MLP_DEPTHS)):
        raise RuntimeError("An MLP dataset has an invalid depth.")
    if np.any(
        ~np.isin(metadata["mlp_hidden_width"][mlp_mask], MLP_HIDDEN_WIDTHS)
    ):
        raise RuntimeError("An MLP dataset has an invalid hidden width.")
    if np.any(~np.isin(metadata["activation"][mlp_mask], list(ACTIVATION_NAMES))):
        raise RuntimeError("An MLP dataset has an invalid activation code.")
    non_mlp_mask = ~mlp_mask
    if np.any(metadata["mlp_depth"][non_mlp_mask] != 0):
        raise RuntimeError("A non-MLP dataset reports an MLP depth.")
    if np.any(metadata["mlp_hidden_width"][non_mlp_mask] != 0):
        raise RuntimeError("A non-MLP dataset reports an MLP hidden width.")
    if np.any(metadata["activation"][non_mlp_mask] != -1):
        raise RuntimeError("A non-MLP dataset reports an activation.")

    nonadditive_with_edges = nonadditive_mask & has_nonroot
    nonadditive_scales = metadata["mean_nonadditive_noise_scale"]
    reported_scales = nonadditive_scales[nonadditive_with_edges]
    if np.any(~np.isfinite(reported_scales)):
        raise RuntimeError(
            "Every non-empty non-additive MLP must report its mean noise scale."
        )
    if np.any(
        (reported_scales < NONADDITIVE_NOISE_SCALE_RANGE[0])
        | (reported_scales > NONADDITIVE_NOISE_SCALE_RANGE[1])
    ):
        raise RuntimeError("A non-additive noise scale is outside its range.")
    if np.any(np.isfinite(nonadditive_scales[~nonadditive_with_edges])):
        raise RuntimeError(
            "Only non-empty non-additive MLPs may report a noise scale."
        )

    if np.any(~np.isin(metadata["rff_num_features"][rff_mask], RFF_FEATURE_COUNTS)):
        raise RuntimeError("An RFF dataset has an invalid feature count.")
    rff_lengthscales = metadata["rff_lengthscale"][rff_mask]
    if np.any(~np.isfinite(rff_lengthscales)):
        raise RuntimeError("An RFF dataset has a non-finite lengthscale.")
    if np.any(
        (rff_lengthscales < RFF_LENGTHSCALE_RANGE[0])
        | (rff_lengthscales > RFF_LENGTHSCALE_RANGE[1])
    ):
        raise RuntimeError("An RFF lengthscale is outside its range.")
    non_rff_mask = ~rff_mask
    if np.any(metadata["rff_num_features"][non_rff_mask] != 0):
        raise RuntimeError("A non-RFF dataset reports an RFF feature count.")
    if np.any(np.isfinite(metadata["rff_lengthscale"][non_rff_mask])):
        raise RuntimeError("A non-RFF dataset reports an RFF lengthscale.")


def generate_dataset(
    *,
    num_nodes: int,
    num_samples: int,
    sem_probs: np.ndarray,
    dataset_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.void]:
    rng = np.random.default_rng(dataset_seed)
    sem_codes = np.asarray(list(SEM_NAMES), dtype=np.int64)
    sem_type = int(rng.choice(sem_codes, p=sem_probs))
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
        details = SemDetails(noise_mode=NOISE_GP_CURRENT)
    else:
        data_topological, details = generate_parametric_sem(
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
    metadata["noise_mode"] = np.int8(details.noise_mode)
    metadata["mean_target_r2"] = np.float32(details.mean_target_r2)
    metadata["mlp_depth"] = np.uint8(details.mlp_depth)
    metadata["mlp_hidden_width"] = np.uint8(details.mlp_hidden_width)
    metadata["activation"] = np.int8(details.activation)
    metadata["mean_nonadditive_noise_scale"] = np.float32(
        details.mean_nonadditive_noise_scale
    )
    metadata["rff_num_features"] = np.uint8(details.rff_num_features)
    metadata["rff_lengthscale"] = np.float32(details.rff_lengthscale)
    return data, dag.astype(np.int8), metadata


def compression_kwargs(name: str) -> Dict[str, object]:
    if name == "none":
        return {}
    if name == "lzf":
        return {"compression": "lzf"}
    if name == "gzip":
        return {"compression": "gzip", "compression_opts": 1}
    raise ValueError(f"Unknown compression: {name}")


def validate_existing_shard(
    *,
    output_path: Path,
    num_nodes: int,
    num_samples: int,
    num_datasets: int,
    sem_probs: np.ndarray,
    shard_seed: int,
) -> None:
    """Ensure a skipped shard belongs to the requested generation run."""
    expected_data_shape = (num_datasets, num_samples, num_nodes)
    expected_label_shape = (num_datasets, num_nodes, num_nodes)
    with h5py.File(output_path, "r") as handle:
        if handle["data"].shape != expected_data_shape:
            raise ValueError(
                f"Existing shard {output_path} has data shape "
                f"{handle['data'].shape}, expected {expected_data_shape}."
            )
        if handle["label"].shape != expected_label_shape:
            raise ValueError(
                f"Existing shard {output_path} has label shape "
                f"{handle['label'].shape}, expected {expected_label_shape}."
            )
        if handle["metadata"].shape != (num_datasets,):
            raise ValueError(f"Existing shard {output_path} has invalid metadata shape.")
        if handle["data"].dtype != np.dtype(np.float32):
            raise ValueError(f"Existing shard {output_path} does not use float32 data.")
        if handle["label"].dtype != np.dtype(np.int8):
            raise ValueError(f"Existing shard {output_path} does not use int8 labels.")
        if handle["metadata"].dtype != METADATA_DTYPE:
            raise ValueError(
                f"Existing shard {output_path} uses a different metadata schema."
            )
        expected_attributes = {
            "distribution": distribution_name(sem_probs),
            "n_nodes": num_nodes,
            "num_samples": num_samples,
            "num_datasets": num_datasets,
            "shard_seed": shard_seed,
            "label_convention": "label[i, j] = 1 means i -> j",
        }
        for name, expected in expected_attributes.items():
            if name not in handle.attrs or handle.attrs[name] != expected:
                raise ValueError(
                    f"Existing shard {output_path} has incompatible attribute "
                    f"{name!r}."
                )
        existing_probabilities = json.loads(handle.attrs["sem_probabilities"])
        requested_probabilities = sem_probability_dict(sem_probs)
        if existing_probabilities.keys() != requested_probabilities.keys() or any(
            not np.isclose(
                float(existing_probabilities[name]),
                requested_probabilities[name],
                rtol=0.0,
                atol=1e-12,
            )
            for name in requested_probabilities
        ):
            raise ValueError(
                f"Existing shard {output_path} uses different SEM probabilities."
            )


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
        validate_existing_shard(
            output_path=output_path,
            num_nodes=num_nodes,
            num_samples=num_samples,
            num_datasets=num_datasets,
            sem_probs=sem_probs,
            shard_seed=shard_seed,
        )
        print(f"[SKIP verified] {output_path}")
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
        handle.attrs["distribution"] = distribution_name(sem_probs)
        handle.attrs["n_nodes"] = int(num_nodes)
        handle.attrs["num_samples"] = int(num_samples)
        handle.attrs["num_datasets"] = int(num_datasets)
        handle.attrs["shard_seed"] = int(shard_seed)
        handle.attrs["label_convention"] = "label[i, j] = 1 means i -> j"
        handle.attrs["sem_type_codes"] = json_compact(SEM_NAMES)
        handle.attrs["graph_type_codes"] = json_compact(GRAPH_NAMES)
        handle.attrs["noise_mode_codes"] = json_compact(NOISE_MODE_NAMES)
        handle.attrs["activation_codes"] = json_compact(ACTIVATION_NAMES)
        handle.attrs["sem_probabilities"] = json_compact(
            sem_probability_dict(sem_probs)
        )

    sem_counts = {
        SEM_NAMES[sem]: int(np.sum(metadata["sem_type"] == sem))
        for sem in SEM_NAMES
    }
    mlp_mask = np.isin(
        metadata["sem_type"],
        [SEM_MLP, SEM_MLP_NONADDITIVE],
    )
    mlp_depth_counts = {
        str(depth): int(np.sum(metadata["mlp_depth"][mlp_mask] == depth))
        for depth in MLP_DEPTHS
    }
    mlp_width_counts = {
        str(width): int(np.sum(metadata["mlp_hidden_width"][mlp_mask] == width))
        for width in MLP_HIDDEN_WIDTHS
    }
    activation_counts = {
        name: int(np.sum(metadata["activation"][mlp_mask] == code))
        for code, name in ACTIVATION_NAMES.items()
    }
    rff_mask = metadata["sem_type"] == SEM_RFF
    rff_feature_counts = {
        str(count): int(np.sum(metadata["rff_num_features"][rff_mask] == count))
        for count in RFF_FEATURE_COUNTS
    }
    nonadditive_scale_summary = finite_summary(
        metadata["mean_nonadditive_noise_scale"]
    )
    rff_lengthscale_summary = finite_summary(metadata["rff_lengthscale"])
    print(
        f"[WRITE] {output_path} data={data.shape} label={labels.shape} "
        f"sem_counts={sem_counts} mlp_depth_counts={mlp_depth_counts} "
        f"mlp_width_counts={mlp_width_counts} "
        f"activation_counts={activation_counts} "
        f"nonadditive_noise_scale={nonadditive_scale_summary} "
        f"rff_feature_counts={rff_feature_counts} "
        f"rff_lengthscale={rff_lengthscale_summary}"
    )


def write_manifest(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    node_counts: Sequence[int],
    node_dataset_counts: Dict[int, int],
    node_count_probabilities: Sequence[float],
    sem_probs: np.ndarray,
    overwrite: bool,
) -> None:
    manifest_path = output_dir / "generation_config.json"
    config = {
        "distribution": distribution_name(sem_probs),
        "split": args.split,
        "node_counts": list(node_counts),
        "num_samples_stored": args.num_samples,
        "training_sample_size_range": [min(100, args.num_samples), args.num_samples],
        "num_datasets_per_node_count": args.num_datasets_per_node_count,
        "num_datasets_total": int(sum(node_dataset_counts.values())),
        "num_datasets_by_node_count": {
            str(node_count): int(node_dataset_counts[node_count])
            for node_count in node_counts
        },
        "requested_node_count_probabilities": {
            str(node_count): float(probability)
            for node_count, probability in zip(
                node_counts,
                node_count_probabilities,
            )
        },
        "realized_node_count_proportions": {
            str(node_count): (
                float(node_dataset_counts[node_count])
                / float(sum(node_dataset_counts.values()))
            )
            for node_count in node_counts
        },
        "datasets_per_file": args.datasets_per_file,
        "seed": args.seed,
        "sem_probabilities": sem_probability_dict(sem_probs),
        "mlp_architecture": {
            "depths": list(MLP_DEPTHS),
            "hidden_widths": list(MLP_HIDDEN_WIDTHS),
            "activations": list(MLP_ACTIVATIONS),
            "leaky_relu_negative_slope": 0.2,
            "sampling_scope": "architecture per dataset; weights per node",
        },
        "mlp_nonadditive": {
            "equation": "X_j = MLP_j(concat(X_pa(j), a_j * Z_j))",
            "noise_scale_distribution": (
                "Uniform("
                f"{NONADDITIVE_NOISE_SCALE_RANGE[0]},"
                f"{NONADDITIVE_NOISE_SCALE_RANGE[1]}) per non-root node"
            ),
            "reported_metadata": "mean_nonadditive_noise_scale",
            "uses_controlled_local_r2": False,
        },
        "rff": {
            "num_features": list(RFF_FEATURE_COUNTS),
            "lengthscale_distribution": (
                "LogUniform("
                f"{RFF_LENGTHSCALE_RANGE[0]},"
                f"{RFF_LENGTHSCALE_RANGE[1]}) per dataset"
            ),
            "frequency_distribution": "Normal(0, lengthscale^-2)",
            "phase_distribution": "Uniform(0, 2*pi)",
            "output_weight_distribution": "Normal(0, 1)",
            "uses_controlled_local_r2": True,
        },
        "graph_probabilities_all_sem_families": {
            "er": 0.5,
            "scale_free": 0.5,
        },
        "edge_count": "UniformInteger(0, min(4*d, d*(d-1)/2))",
        "sample_statement": (
            "Rows share one sampled SCM and independent raw exogenous-noise draws. "
            "Additive SEMs use article-style empirical R2 calibration. All "
            "SEM families use node standardisation, making saved rows "
            "exchangeable rather than strictly independent."
        ),
        "hdf5_datasets": {
            "data": ["datasets_in_shard", args.num_samples, "num_nodes"],
            "label": ["datasets_in_shard", "num_nodes", "num_nodes"],
            "metadata": ["datasets_in_shard"],
        },
        "metadata_fields": list(METADATA_DTYPE.names or ()),
        "metadata_sentinels": {
            "not_applicable_integer": 0,
            "not_applicable_activation": -1,
            "not_applicable_float": "NaN",
        },
        "compression": args.compression,
    }
    if manifest_path.exists() and not overwrite:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            existing_config = json.load(handle)
        if existing_config != config:
            raise ValueError(
                f"Existing manifest {manifest_path} does not match the requested "
                "generation configuration. Use a different --save_root, or pass "
                "--overwrite only when replacement is intentional."
            )
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed-shape HDF5 shards containing mixed GP, linear, "
            "additive MLP, non-additive MLP, and RFF causal datasets."
        )
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=(
            "ml2_meta_causal_discovery/datasets/data/synth_training_data/"
            "diverse_causal_pretrain_v2"
        ),
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--node_counts", type=str, default="3,4,5")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument(
        "--num_datasets_per_node_count",
        type=int,
        default=None,
        help="Generate this many datasets for every requested node count.",
    )
    parser.add_argument(
        "--num_datasets_total",
        type=int,
        default=None,
        help=(
            "Total datasets across node counts. Use instead of "
            "--num_datasets_per_node_count."
        ),
    )
    parser.add_argument(
        "--node_count_probabilities",
        type=str,
        default=None,
        help=(
            "Comma-separated probabilities aligned with --node_counts. Requires "
            "--num_datasets_total; defaults to uniform when omitted."
        ),
    )
    parser.add_argument("--datasets_per_file", type=int, default=500)
    parser.add_argument("--gp_probability", type=float, default=0.35)
    parser.add_argument(
        "--mlp_probability",
        "--mlp_additive_probability",
        dest="mlp_probability",
        type=float,
        default=0.50,
        help="Probability of the additive MLP SEM (legacy name: mlp).",
    )
    parser.add_argument(
        "--linear_probability",
        "--linear_additive_probability",
        dest="linear_probability",
        type=float,
        default=0.15,
        help="Probability of the additive linear SEM.",
    )
    parser.add_argument("--mlp_nonadditive_probability", type=float, default=0.0)
    parser.add_argument("--rff_probability", type=float, default=0.0)
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
    if len(set(node_counts)) != len(node_counts):
        raise ValueError("--node_counts must not contain duplicates.")
    if args.num_samples <= 1:
        raise ValueError("--num_samples must be greater than 1.")
    if args.datasets_per_file <= 0:
        raise ValueError("--datasets_per_file must be positive.")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")

    sem_probs = validate_probabilities(
        args.gp_probability,
        args.mlp_probability,
        args.linear_probability,
        args.mlp_nonadditive_probability,
        args.rff_probability,
    )
    node_dataset_counts, node_count_probabilities = resolve_node_dataset_counts(
        args,
        node_counts,
    )
    output_dir = Path(args.save_root).expanduser().resolve() / args.split
    write_manifest(
        output_dir=output_dir,
        args=args,
        node_counts=node_counts,
        node_dataset_counts=node_dataset_counts,
        node_count_probabilities=node_count_probabilities,
        sem_probs=sem_probs,
        overwrite=args.overwrite,
    )

    master_rng = np.random.default_rng(args.seed)
    print("=" * 100)
    print("Generating diverse causal pretraining data")
    print(f"output_dir:                  {output_dir}")
    print(f"node_counts:                 {node_counts}")
    print(f"num_samples per dataset:     {args.num_samples}")
    print(f"datasets by node count:      {node_dataset_counts}")
    print(f"total datasets:              {sum(node_dataset_counts.values())}")
    print(f"datasets per HDF5:           {args.datasets_per_file}")
    print(f"parallel shard workers:      {args.workers}")
    print(
        "SEM probabilities:          "
        + ", ".join(
            f"{name}={probability:.3f}"
            for name, probability in sem_probability_dict(sem_probs).items()
        )
    )
    print(f"compression:                 {args.compression}")
    print("=" * 100)

    shard_jobs: List[Dict[str, Any]] = []
    for num_nodes in node_counts:
        num_datasets_for_node_count = node_dataset_counts[num_nodes]
        num_parts = math.ceil(
            num_datasets_for_node_count / args.datasets_per_file
        )
        for part_index in range(num_parts):
            start = part_index * args.datasets_per_file
            end = min(
                num_datasets_for_node_count,
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
        for job in tqdm(shard_jobs, desc="HDF5 shards", unit="shard"):
            write_shard(**job)
        return

    # "spawn" avoids forking an already-initialised TensorFlow runtime.
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
    ) as executor:
        futures = [executor.submit(write_shard, **job) for job in shard_jobs]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="HDF5 shards",
            unit="shard",
        ):
            future.result()


if __name__ == "__main__":
    main()
