#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


N_NODES = 3
EDGE_ORDER: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
]


# =========================
# Small self-contained distributions
# =========================

class Distribution:
    def __call__(self, rng: np.random.Generator, shape=None):
        raise NotImplementedError


@dataclass
class Gaussian(Distribution):
    scale: float = 1.0

    def __call__(self, rng: np.random.Generator, shape=None):
        return self.scale * rng.normal(size=shape)


@dataclass
class Laplace(Distribution):
    scale: float = 1.0

    def __call__(self, rng: np.random.Generator, shape=None):
        return self.scale * rng.laplace(size=shape)


@dataclass
class Cauchy(Distribution):
    scale: float = 1.0

    def __call__(self, rng: np.random.Generator, shape=None):
        return self.scale * rng.standard_cauchy(size=shape)


@dataclass
class Uniform(Distribution):
    low: float
    high: float

    def __call__(self, rng: np.random.Generator, shape=None):
        return rng.uniform(low=self.low, high=self.high, size=shape)


@dataclass
class SignedUniform(Distribution):
    low: float
    high: float

    def __call__(self, rng: np.random.Generator, shape=None):
        signs = rng.choice(np.array([-1.0, 1.0]), size=shape)
        mags = rng.uniform(low=self.low, high=self.high, size=shape)
        return signs * mags


# =========================
# Graph helpers
# =========================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed-graph 3-node benchmark datasets for AVICI/traditional-method evaluation."
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="/home/rtyen/projects/avici-main",
        help="Root directory of the avici-main project.",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help="Root directory for saving generated benchmark data. Default: <work_dir>/benchmark_data",
    )
    parser.add_argument(
        "--graph_id",
        type=int,
        default=None,
        help="Graph id in [0, 24] for the fixed 3-node DAG to generate.",
    )
    parser.add_argument(
        "--mechanism",
        type=str,
        default="linear",
        choices=["linear", "gp", "rff", "nonlinear"],
        help="Mechanism family. 'gp' and 'nonlinear' are aliases of 'rff'.",
    )
    parser.add_argument(
        "--noise",
        type=str,
        default="gaussian",
        choices=["gaussian", "laplace", "cauchy", "gaussian_heteroskedastic"],
        help="Additive noise family.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of observational samples per dataset.",
    )
    parser.add_argument(
        "--num_datasets",
        type=int,
        default=100,
        help="How many independently generated datasets to save in the HDF5 file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Master random seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file if it already exists.",
    )
    parser.add_argument(
        "--list_graphs",
        action="store_true",
        help="Print all 25 three-node DAG ids and exit.",
    )
    return parser.parse_args()



def is_acyclic(adj: np.ndarray) -> bool:
    d = adj.shape[0]
    indeg = adj.sum(axis=0).astype(int)
    queue = [i for i in range(d) if indeg[i] == 0]
    visited = 0

    while queue:
        u = queue.pop(0)
        visited += 1
        for v in np.where(adj[u] == 1)[0]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(int(v))
    return visited == d



def topological_order(adj: np.ndarray) -> List[int]:
    d = adj.shape[0]
    indeg = adj.sum(axis=0).astype(int)
    queue = [i for i in range(d) if indeg[i] == 0]
    order: List[int] = []

    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in np.where(adj[u] == 1)[0]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(int(v))

    if len(order) != d:
        raise ValueError("Adjacency matrix is not a DAG.")
    return order



def graph_bitstring(adj: np.ndarray) -> str:
    return "".join(str(int(adj[i, j])) for (i, j) in EDGE_ORDER)



def graph_name_from_adj(adj: np.ndarray) -> str:
    edges = [f"{i}to{j}" for (i, j) in EDGE_ORDER if adj[i, j] == 1]
    return "empty" if not edges else "__".join(edges)



def enumerate_three_node_dags() -> List[Dict]:
    dags: List[Dict] = []
    for mask in range(1 << len(EDGE_ORDER)):
        adj = np.zeros((N_NODES, N_NODES), dtype=np.int8)
        for bit_idx, (i, j) in enumerate(EDGE_ORDER):
            if (mask >> bit_idx) & 1:
                adj[i, j] = 1
        if is_acyclic(adj):
            dags.append(
                {
                    "adjacency": adj,
                    "edge_count": int(adj.sum()),
                    "bitstring": graph_bitstring(adj),
                    "name": graph_name_from_adj(adj),
                }
            )

    dags.sort(key=lambda item: (item["edge_count"], item["bitstring"]))
    for idx, item in enumerate(dags):
        item["graph_id"] = idx
    return dags


# =========================
# Mechanisms and noise
# =========================


def canonicalize_mechanism(name: str) -> str:
    mapping = {
        "linear": "linear",
        "gp": "rff",
        "rff": "rff",
        "nonlinear": "rff",
    }
    return mapping[name]



def build_noise_components(noise_name: str):
    if noise_name == "gaussian":
        return Gaussian(), Uniform(low=0.2, high=2.0), None
    if noise_name == "laplace":
        return Laplace(), Uniform(low=0.2, high=2.0), None
    if noise_name == "cauchy":
        return Cauchy(), Uniform(low=0.2, high=2.0), None
    if noise_name == "gaussian_heteroskedastic":
        return Gaussian(), None, {"length_scale": 10.0, "output_scale": 2.0}
    raise ValueError(f"Unsupported noise: {noise_name}")



def draw_rff_params(
    rng: np.random.Generator,
    d: int,
    length_scale: Distribution | float,
    output_scale: Distribution | float,
    n_rff: int,
):
    ls = length_scale(rng, shape=(1,)).item() if callable(length_scale) else float(length_scale)
    c = output_scale(rng, shape=(1,)).item() if callable(output_scale) else float(output_scale)
    omega = rng.normal(loc=0.0, scale=1.0 / ls, size=(d, n_rff))
    b = rng.uniform(low=0.0, high=2 * np.pi, size=(n_rff,))
    w = rng.normal(loc=0.0, scale=1.0, size=(n_rff,))
    return {
        "length_scale_value": ls,
        "output_scale_value": c,
        "omega": omega,
        "b": b,
        "w": w,
        "n_rff": n_rff,
    }


class SimpleNoise:
    def __init__(self, dist: Distribution, scale: float):
        self.dist = dist
        self.scale = scale

    def __call__(self, rng: np.random.Generator, x: np.ndarray, is_parent: np.ndarray) -> np.ndarray:
        return self.scale * self.dist(rng, shape=(x.shape[0],))


class HeteroscedasticRFFNoise:
    def __init__(self, dist: Distribution, rng: np.random.Generator, d: int, length_scale: float, output_scale: float, n_rff: int = 100):
        self.dist = dist
        self.param = draw_rff_params(
            rng=rng,
            d=d,
            length_scale=length_scale,
            output_scale=output_scale,
            n_rff=n_rff,
        )
        self.n_rff = n_rff

    def __call__(self, rng: np.random.Generator, x: np.ndarray, is_parent: np.ndarray) -> np.ndarray:
        parent_idx = np.where(is_parent)[0]
        if len(parent_idx) == 0:
            f_x = np.zeros((x.shape[0],), dtype=np.float64)
        else:
            x_parents = x[:, parent_idx]
            phi = np.cos(np.einsum("db,nd->nb", self.param["omega"], x_parents) + self.param["b"])
            f_x = np.sqrt(2.0) * self.param["output_scale_value"] * np.einsum("b,nb->n", self.param["w"], phi) / np.sqrt(self.n_rff)
        scale = np.sqrt(np.log1p(np.exp(f_x)))
        return scale * self.dist(rng, shape=scale.shape)



def init_noise_dist(
    rng: np.random.Generator,
    dim: int,
    dist: Distribution,
    noise_scale: Optional[Distribution],
    noise_scale_heteroskedastic: Optional[Dict],
):
    if noise_scale is not None:
        scale = float(noise_scale(rng, shape=(1,)).item())
        return SimpleNoise(dist=dist, scale=scale)
    if noise_scale_heteroskedastic is not None:
        return HeteroscedasticRFFNoise(
            dist=dist,
            rng=rng,
            d=int(dim),
            length_scale=float(noise_scale_heteroskedastic["length_scale"]),
            output_scale=float(noise_scale_heteroskedastic["output_scale"]),
            n_rff=100,
        )
    raise ValueError("Either noise_scale or noise_scale_heteroskedastic must be provided.")


class LinearSCMGenerator:
    def __init__(self, noise_name: str):
        noise_dist, noise_scale, noise_scale_heteroskedastic = build_noise_components(noise_name)
        self.param = SignedUniform(low=1.0, high=3.0)
        self.bias = Uniform(low=-3.0, high=3.0)
        self.noise_dist = noise_dist
        self.noise_scale = noise_scale
        self.noise_scale_heteroskedastic = noise_scale_heteroskedastic

    def sample(self, rng: np.random.Generator, g: np.ndarray, n: int) -> Tuple[np.ndarray, Dict]:
        d = g.shape[0]
        order = topological_order(g)
        x = np.zeros((n, d), dtype=np.float64)
        node_params: List[Dict] = []

        mechanisms = []
        noise_models = []
        for j in range(d):
            w = self.param(rng, shape=(d,)).astype(np.float64)
            b = float(self.bias(rng, shape=(1,)).item())
            mechanisms.append((w, b))
            noise_models.append(
                init_noise_dist(
                    rng=rng,
                    dim=int(g[:, j].sum()),
                    dist=self.noise_dist,
                    noise_scale=self.noise_scale,
                    noise_scale_heteroskedastic=self.noise_scale_heteroskedastic,
                )
            )
            node_params.append(
                {
                    "node": j,
                    "weights": w.tolist(),
                    "bias": b,
                    "n_parents": int(g[:, j].sum()),
                }
            )

        for j in order:
            is_parent = g[:, j].astype(bool)
            z_j = noise_models[j](rng=rng, x=x, is_parent=is_parent)
            w_j, b_j = mechanisms[j]
            x[:, j] = x @ (w_j * is_parent.astype(np.float64)) + b_j + z_j

        meta = {
            "mechanism": "linear",
            "node_params": node_params,
        }
        return x.astype(np.float32), meta


class RFFSCMGenerator:
    def __init__(self, noise_name: str):
        noise_dist, noise_scale, noise_scale_heteroskedastic = build_noise_components(noise_name)
        self.length_scale = Uniform(low=7.0, high=10.0)
        self.output_scale = Uniform(low=10.0, high=20.0)
        self.bias = Uniform(low=-3.0, high=3.0)
        self.noise_dist = noise_dist
        self.noise_scale = noise_scale
        self.noise_scale_heteroskedastic = noise_scale_heteroskedastic

    def sample(self, rng: np.random.Generator, g: np.ndarray, n: int) -> Tuple[np.ndarray, Dict]:
        d = g.shape[0]
        order = topological_order(g)
        x = np.zeros((n, d), dtype=np.float64)
        node_params: List[Dict] = []

        mechanisms = []
        noise_models = []
        for j in range(d):
            n_parents = int(g[:, j].sum())
            rff_param = draw_rff_params(
                rng=rng,
                d=n_parents,
                length_scale=self.length_scale,
                output_scale=self.output_scale,
                n_rff=100,
            )
            b_j = float(self.bias(rng, shape=(1,)).item())
            mechanisms.append((rff_param, b_j))
            noise_models.append(
                init_noise_dist(
                    rng=rng,
                    dim=n_parents,
                    dist=self.noise_dist,
                    noise_scale=self.noise_scale,
                    noise_scale_heteroskedastic=self.noise_scale_heteroskedastic,
                )
            )
            node_params.append(
                {
                    "node": j,
                    "bias": b_j,
                    "n_parents": n_parents,
                    "length_scale": rff_param["length_scale_value"],
                    "output_scale": rff_param["output_scale_value"],
                    "n_rff": int(rff_param["n_rff"]),
                }
            )

        for j in order:
            is_parent = g[:, j].astype(bool)
            z_j = noise_models[j](rng=rng, x=x, is_parent=is_parent)
            parent_idx = np.where(is_parent)[0]
            rff_param, b_j = mechanisms[j]
            if len(parent_idx) == 0:
                f_j = np.zeros((n,), dtype=np.float64)
            else:
                x_parents = x[:, parent_idx]
                phi = np.cos(np.einsum("db,nd->nb", rff_param["omega"], x_parents) + rff_param["b"])
                f_j = (
                    np.sqrt(2.0)
                    * rff_param["output_scale_value"]
                    * np.einsum("b,nb->n", rff_param["w"], phi)
                    / np.sqrt(rff_param["n_rff"])
                )
            x[:, j] = f_j + b_j + z_j

        meta = {
            "mechanism": "rff",
            "node_params": node_params,
        }
        return x.astype(np.float32), meta



def build_generator(mechanism_name: str, noise_name: str):
    mechanism_name = canonicalize_mechanism(mechanism_name)
    if mechanism_name == "linear":
        return LinearSCMGenerator(noise_name=noise_name), mechanism_name
    if mechanism_name == "rff":
        return RFFSCMGenerator(noise_name=noise_name), mechanism_name
    raise ValueError(f"Unsupported mechanism: {mechanism_name}")


# =========================
# I/O helpers
# =========================


def pretty_graph_listing(dags: List[Dict]) -> str:
    lines = []
    for item in dags:
        gid = item["graph_id"]
        name = item["name"]
        adj = item["adjacency"]
        lines.append(f"graph_id={gid:02d}  name={name}")
        lines.append(str(adj))
        lines.append("")
    return "\n".join(lines)



def build_output_dir(
    save_root: Path,
    graph_info: Dict,
    mechanism_name: str,
    noise_name: str,
    num_samples: int,
    seed: int,
) -> Path:
    graph_dir = f"graph_{graph_info['graph_id']:02d}__{graph_info['name']}"
    return (
        save_root
        / str(N_NODES)
        / graph_dir
        / mechanism_name
        / noise_name
        / f"n_{num_samples}"
        / f"seed_{seed}"
    )


# =========================
# Main
# =========================


def main() -> None:
    args = parse_args()
    dags = enumerate_three_node_dags()

    if args.list_graphs:
        print(pretty_graph_listing(dags))
        return

    if args.graph_id is None:
        raise ValueError("--graph_id is required unless you use --list_graphs")
    if not (0 <= args.graph_id < len(dags)):
        raise ValueError(f"graph_id must be in [0, {len(dags) - 1}], got {args.graph_id}")
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive")
    if args.num_datasets <= 0:
        raise ValueError("--num_datasets must be positive")

    work_dir = Path(args.work_dir).expanduser().resolve()
    save_root = Path(args.save_root).expanduser().resolve() if args.save_root else (work_dir / "benchmark_data")

    graph_info = dags[args.graph_id]
    adjacency = graph_info["adjacency"].astype(np.int8)
    generator, mechanism_name = build_generator(args.mechanism, args.noise)
    noise_name = args.noise

    output_dir = build_output_dir(
        save_root=save_root,
        graph_info=graph_info,
        mechanism_name=mechanism_name,
        noise_name=noise_name,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = output_dir / f"benchmark_3var_numdatasets_{args.num_datasets}.h5"
    config_path = output_dir / "config.json"
    dataset_meta_path = output_dir / "dataset_meta.json"

    if h5_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {h5_path}\n"
            "Use --overwrite if you want to replace it."
        )

    master_rng = np.random.default_rng(args.seed)
    dataset_seeds = master_rng.integers(0, np.iinfo(np.uint32).max, size=args.num_datasets, dtype=np.uint32)

    all_data = np.zeros((args.num_datasets, args.num_samples, N_NODES), dtype=np.float32)
    all_interv = np.zeros((args.num_datasets, args.num_samples, N_NODES), dtype=np.float32)
    all_labels = np.repeat(adjacency[None, :, :], repeats=args.num_datasets, axis=0).astype(np.int8)
    dataset_meta: List[Dict] = []

    for dataset_idx, dataset_seed in enumerate(dataset_seeds.tolist()):
        rng = np.random.default_rng(int(dataset_seed))
        x_obs, meta = generator.sample(rng=rng, g=adjacency, n=args.num_samples)
        all_data[dataset_idx] = x_obs
        dataset_meta.append(
            {
                "dataset_idx": dataset_idx,
                "dataset_seed": int(dataset_seed),
                **meta,
            }
        )

    config = {
        "project_root": str(work_dir),
        "save_root": str(save_root),
        "n_nodes": N_NODES,
        "graph_id": int(graph_info["graph_id"]),
        "graph_name": graph_info["name"],
        "graph_bitstring": graph_info["bitstring"],
        "adjacency": adjacency.astype(int).tolist(),
        "mechanism": mechanism_name,
        "noise": noise_name,
        "num_samples": int(args.num_samples),
        "num_datasets": int(args.num_datasets),
        "seed": int(args.seed),
        "observational_only": True,
        "datasets_in_h5": {
            "data": [args.num_datasets, args.num_samples, N_NODES],
            "interv": [args.num_datasets, args.num_samples, N_NODES],
            "label": [args.num_datasets, N_NODES, N_NODES],
            "dataset_seed": [args.num_datasets],
        },
        "notes": {
            "data": "observational samples only, shape [num_datasets, num_samples, 3]",
            "interv": "intervention mask aligned with data; currently all zeros because this script only generates observational data",
            "label": "true adjacency matrix, shape [num_datasets, 3, 3], with label[i, j] = 1 meaning i -> j",
            "dataset_meta_json": "per-dataset sampled mechanism parameters for reproducibility/inspection",
        },
        "recommended_path_pattern": "benchmark_data/3/graph_xx__name/mechanism/noise/n_<num_samples>/seed_<seed>/",
    }

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("data", data=all_data, compression="gzip")
        f.create_dataset("interv", data=all_interv, compression="gzip")
        f.create_dataset("label", data=all_labels, compression="gzip")
        f.create_dataset("dataset_seed", data=dataset_seeds.astype(np.uint32), compression="gzip")

        f.attrs["n_nodes"] = N_NODES
        f.attrs["graph_id"] = int(graph_info["graph_id"])
        f.attrs["graph_name"] = graph_info["name"]
        f.attrs["graph_bitstring"] = graph_info["bitstring"]
        f.attrs["mechanism"] = mechanism_name
        f.attrs["noise"] = noise_name
        f.attrs["num_samples"] = int(args.num_samples)
        f.attrs["num_datasets"] = int(args.num_datasets)
        f.attrs["seed"] = int(args.seed)
        f.attrs["config_json"] = json.dumps(config, ensure_ascii=False)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with open(dataset_meta_path, "w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Finished generating benchmark datasets.")
    print(f"Output directory : {output_dir}")
    print(f"HDF5 file        : {h5_path}")
    print(f"Config file      : {config_path}")
    print(f"Dataset meta     : {dataset_meta_path}")
    print("Graph info:")
    print(f"  graph_id       : {graph_info['graph_id']}")
    print(f"  graph_name     : {graph_info['name']}")
    print(f"  mechanism      : {mechanism_name}")
    print(f"  noise          : {noise_name}")
    print(f"  num_samples    : {args.num_samples}")
    print(f"  num_datasets   : {args.num_datasets}")
    print(f"  seed           : {args.seed}")
    print("Adjacency:")
    print(adjacency)
    print("=" * 80)


if __name__ == "__main__":
    main()
