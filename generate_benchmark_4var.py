#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import tensorflow as tf

from ml2_meta_causal_discovery.datasets.functions_generator import GPFunctionGenerator


N_NODES = 4
EDGE_ORDER: List[Tuple[int, int]] = [
    (i, j) for i in range(N_NODES) for j in range(N_NODES) if i != j
]


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


def json_dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


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
    return "".join(str(int(adj[i, j])) for i, j in EDGE_ORDER)


def graph_name_from_adj(adj: np.ndarray) -> str:
    edges = [f"{i}to{j}" for i, j in EDGE_ORDER if adj[i, j] == 1]
    return "empty" if not edges else "__".join(edges)


def permute_adj(adj: np.ndarray, perm: Tuple[int, ...]) -> np.ndarray:
    perm_arr = np.asarray(perm, dtype=int)
    return adj[perm_arr, :][:, perm_arr]


def canonical_dag(adj: np.ndarray) -> Tuple[str, np.ndarray]:
    best_key: Optional[str] = None
    best_adj: Optional[np.ndarray] = None
    for perm in _all_perms(adj.shape[0]):
        curr = permute_adj(adj, perm)
        key = graph_bitstring(curr)
        if best_key is None or key < best_key:
            best_key = key
            best_adj = curr
    assert best_key is not None and best_adj is not None
    return best_key, best_adj.astype(np.int8)


def _all_perms(n: int) -> List[Tuple[int, ...]]:
    if n != 4:
        import itertools

        return list(itertools.permutations(range(n)))
    return [
        (0, 1, 2, 3),
        (0, 1, 3, 2),
        (0, 2, 1, 3),
        (0, 2, 3, 1),
        (0, 3, 1, 2),
        (0, 3, 2, 1),
        (1, 0, 2, 3),
        (1, 0, 3, 2),
        (1, 2, 0, 3),
        (1, 2, 3, 0),
        (1, 3, 0, 2),
        (1, 3, 2, 0),
        (2, 0, 1, 3),
        (2, 0, 3, 1),
        (2, 1, 0, 3),
        (2, 1, 3, 0),
        (2, 3, 0, 1),
        (2, 3, 1, 0),
        (3, 0, 1, 2),
        (3, 0, 2, 1),
        (3, 1, 0, 2),
        (3, 1, 2, 0),
        (3, 2, 0, 1),
        (3, 2, 1, 0),
    ]


def enumerate_nonisomorphic_four_node_dags() -> List[Dict[str, Any]]:
    reps: Dict[str, Dict[str, Any]] = {}
    for mask in range(1 << len(EDGE_ORDER)):
        adj = np.zeros((N_NODES, N_NODES), dtype=np.int8)
        for bit_idx, (i, j) in enumerate(EDGE_ORDER):
            if (mask >> bit_idx) & 1:
                adj[i, j] = 1
        if not is_acyclic(adj):
            continue
        key, canon_adj = canonical_dag(adj)
        if key not in reps:
            reps[key] = {
                "adjacency": canon_adj,
                "canonical_key": key,
                "edge_count": int(canon_adj.sum()),
                "bitstring": graph_bitstring(canon_adj),
                "name": graph_name_from_adj(canon_adj),
            }

    dags = sorted(reps.values(), key=lambda item: (item["edge_count"], item["bitstring"]))
    for idx, item in enumerate(dags):
        item["graph_id"] = idx
    return dags


def parse_graph_json(s: str) -> Optional[np.ndarray]:
    if not s:
        return None
    adj = np.asarray(json.loads(s), dtype=np.int8)
    if adj.shape != (N_NODES, N_NODES):
        raise ValueError(f"--graph_json must have shape {(N_NODES, N_NODES)}, got {adj.shape}.")
    np.fill_diagonal(adj, 0)
    if not is_acyclic(adj):
        raise ValueError("--graph_json must be a DAG.")
    return adj


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
    def __init__(
        self,
        dist: Distribution,
        rng: np.random.Generator,
        d: int,
        length_scale: float,
        output_scale: float,
        n_rff: int = 100,
    ):
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
            f_x = (
                np.sqrt(2.0)
                * self.param["output_scale_value"]
                * np.einsum("b,nb->n", self.param["w"], phi)
                / np.sqrt(self.n_rff)
            )
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
    mechanism_name = "linear"

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
            node_params.append({"node": j, "weights": w.tolist(), "bias": b, "n_parents": int(g[:, j].sum())})

        for j in order:
            is_parent = g[:, j].astype(bool)
            z_j = noise_models[j](rng=rng, x=x, is_parent=is_parent)
            w_j, b_j = mechanisms[j]
            x[:, j] = x @ (w_j * is_parent.astype(np.float64)) + b_j + z_j

        return x.astype(np.float32), {"mechanism": self.mechanism_name, "node_params": node_params}


class RFFSCMGenerator:
    mechanism_name = "rff"

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

        return x.astype(np.float32), {"mechanism": self.mechanism_name, "node_params": node_params}


class CSNPGPSCMGenerator:
    mechanism_name = "csnp_gp"

    def __init__(self):
        self.generator = GPFunctionGenerator(
            num_variables=N_NODES,
            num_samples=1,
            interventions=False,
        )

    def sample(self, rng: np.random.Generator, g: np.ndarray, n: int) -> Tuple[np.ndarray, Dict]:
        del rng
        old_num_samples = self.generator.num_samples
        self.generator.num_samples = n
        order = topological_order(g)
        topo_g = g[np.ix_(order, order)]
        x_topo = self.generator.generate_data(causal_graph=topo_g, num_int_samples=n)
        self.generator.num_samples = old_num_samples
        x = np.zeros_like(x_topo)
        x[:, np.asarray(order, dtype=int)] = x_topo
        return x.astype(np.float32), {"mechanism": self.mechanism_name, "topological_order_used": order}


def build_generator(generator_name: str, noise_name: str):
    if generator_name == "linear":
        return LinearSCMGenerator(noise_name=noise_name), "linear", noise_name
    if generator_name in {"rff", "nonlinear", "gp"}:
        return RFFSCMGenerator(noise_name=noise_name), "rff", noise_name
    if generator_name == "csnp_gp":
        return CSNPGPSCMGenerator(), "csnp_gp", "csnp_gp_default"
    raise ValueError(f"Unsupported generator: {generator_name}")


def pretty_graph_listing(dags: List[Dict[str, Any]]) -> str:
    lines = []
    for item in dags:
        lines.append(
            f"graph_id={item['graph_id']:02d} edge_count={item['edge_count']} "
            f"name={item['name']} canonical_key={item['canonical_key']}"
        )
        lines.append(str(item["adjacency"]))
        lines.append("")
    return "\n".join(lines)


def graph_info_from_json(adj: np.ndarray, dags: List[Dict[str, Any]]) -> Dict[str, Any]:
    key, canon_adj = canonical_dag(adj)
    match = next((item for item in dags if item["canonical_key"] == key), None)
    graph_id = -1 if match is None else int(match["graph_id"])
    return {
        "graph_id": graph_id,
        "edge_count": int(adj.sum()),
        "adjacency": adj.astype(np.int8),
        "canonical_key": key,
        "canonical_adjacency": canon_adj.astype(int).tolist(),
        "bitstring": graph_bitstring(adj),
        "name": graph_name_from_adj(adj),
    }


def select_graphs(args: argparse.Namespace, dags: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    graph_json = parse_graph_json(args.graph_json)
    if graph_json is not None:
        return [graph_info_from_json(graph_json, dags)]
    graph_ids = parse_int_list(args.graph_ids)
    if not graph_ids:
        return dags
    selected = []
    for graph_id in graph_ids:
        if not (0 <= graph_id < len(dags)):
            raise ValueError(f"graph_id must be in [0, {len(dags) - 1}], got {graph_id}")
        selected.append(dags[graph_id])
    return selected


def build_output_dir(
    save_root: Path,
    graph_info: Dict[str, Any],
    generator_name: str,
    mechanism_name: str,
    noise_name: str,
    num_samples: int,
    seed: int,
) -> Path:
    graph_dir = f"graph_{int(graph_info['graph_id']):02d}__{graph_info['name']}"
    if generator_name == "csnp_gp":
        variant_name = "csnp_gp"
    else:
        variant_name = f"{mechanism_name}_{noise_name}"
    return (
        save_root
        / str(N_NODES)
        / graph_dir
        / variant_name
        / f"n_{num_samples}"
        / f"seed_{seed}"
    )


def generate_h5(
    *,
    h5_path: Path,
    config_path: Path,
    dataset_meta_path: Path,
    graph_info: Dict[str, Any],
    generator_name: str,
    mechanism_name: str,
    noise_name: str,
    num_samples: int,
    num_datasets: int,
    seed: int,
    overwrite: bool,
) -> None:
    if h5_path.exists() and not overwrite:
        print(f"[SKIP] exists: {h5_path}")
        return

    generator, mechanism_name, noise_name = build_generator(generator_name, noise_name)
    adjacency = graph_info["adjacency"].astype(np.int8)
    master_rng = np.random.default_rng(seed)
    dataset_seeds = master_rng.integers(0, np.iinfo(np.uint32).max, size=num_datasets, dtype=np.uint32)

    all_data = np.zeros((num_datasets, num_samples, N_NODES), dtype=np.float32)
    all_labels = np.repeat(adjacency[None, :, :], repeats=num_datasets, axis=0).astype(np.int8)
    dataset_meta: List[Dict[str, Any]] = []

    for dataset_idx, dataset_seed in enumerate(dataset_seeds.tolist()):
        rng = np.random.default_rng(int(dataset_seed))
        np.random.seed(int(dataset_seed))
        tf.random.set_seed(int(dataset_seed))
        x_obs, meta = generator.sample(rng=rng, g=adjacency, n=num_samples)
        all_data[dataset_idx] = x_obs
        dataset_meta.append({"dataset_idx": dataset_idx, "dataset_seed": int(dataset_seed), **meta})

    config = {
        "n_nodes": N_NODES,
        "distribution": "fixed_graph_benchmark_4var",
        "graph_id": int(graph_info["graph_id"]),
        "graph_name": graph_info["name"],
        "graph_bitstring": graph_info["bitstring"],
        "canonical_key": graph_info["canonical_key"],
        "edge_count": int(graph_info["edge_count"]),
        "adjacency": adjacency.astype(int).tolist(),
        "generator": generator_name,
        "mechanism": mechanism_name,
        "noise": noise_name,
        "num_samples": int(num_samples),
        "num_datasets": int(num_datasets),
        "seed": int(seed),
        "observational_only": True,
        "datasets_in_h5": {
            "data": [num_datasets, num_samples, N_NODES],
            "label": [num_datasets, N_NODES, N_NODES],
            "dataset_seed": [num_datasets],
        },
        "notes": {
            "label": "true adjacency matrix with label[i, j] = 1 meaning i -> j",
            "graph_selection": "graph_id indexes non-isomorphic 4-node DAG representatives.",
        },
    }

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("data", data=all_data, compression="gzip")
        f.create_dataset("label", data=all_labels, compression="gzip")
        f.create_dataset("dataset_seed", data=dataset_seeds.astype(np.uint32), compression="gzip")
        f.attrs["n_nodes"] = N_NODES
        f.attrs["distribution"] = "fixed_graph_benchmark_4var"
        f.attrs["graph_id"] = int(graph_info["graph_id"])
        f.attrs["graph_name"] = graph_info["name"]
        f.attrs["graph_bitstring"] = graph_info["bitstring"]
        f.attrs["canonical_key"] = graph_info["canonical_key"]
        f.attrs["edge_count"] = int(graph_info["edge_count"])
        f.attrs["generator"] = generator_name
        f.attrs["mechanism"] = mechanism_name
        f.attrs["noise"] = noise_name
        f.attrs["num_samples"] = int(num_samples)
        f.attrs["num_datasets"] = int(num_datasets)
        f.attrs["seed"] = int(seed)
        f.attrs["label_convention"] = "label[i, j] = 1 means i -> j"
        f.attrs["config_json"] = json_dumps(config)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(dataset_meta_path, "w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, indent=2, ensure_ascii=False)

    print(f"[WRITE] {h5_path} data={tuple(all_data.shape)} label={tuple(all_labels.shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed-graph 4-node benchmark datasets."
    )
    parser.add_argument("--save_root", type=str, default="benchmark_data_4var_fixed")
    parser.add_argument(
        "--generator",
        type=str,
        default="csnp_gp",
        help=(
            "Comma list of data mechanism families. Supported: csnp_gp, linear, rff, gp, nonlinear. "
            "gp/nonlinear are aliases of rff here; csnp_gp uses CSNP's GPFunctionGenerator."
        ),
    )
    parser.add_argument(
        "--noise",
        type=str,
        default="gaussian",
        choices=["gaussian", "laplace", "cauchy", "gaussian_heteroskedastic"],
        help="Noise family for linear/rff. Ignored by csnp_gp.",
    )
    parser.add_argument("--sample_sizes", type=str, default="5,20,50,100,300,1000,3000")
    parser.add_argument("--num_datasets", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--graph_ids",
        type=str,
        default="",
        help="Comma list of graph ids. Empty means all non-isomorphic representatives.",
    )
    parser.add_argument(
        "--graph_json",
        type=str,
        default="",
        help="Optional explicit 4x4 adjacency JSON. Overrides --graph_ids.",
    )
    parser.add_argument("--list_graphs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dags = enumerate_nonisomorphic_four_node_dags()
    if args.list_graphs:
        print(pretty_graph_listing(dags))
        print(f"num_graphs={len(dags)}")
        return

    sample_sizes = parse_int_list(args.sample_sizes)
    generators = parse_str_list(args.generator)
    selected_graphs = select_graphs(args, dags)
    save_root = Path(args.save_root).expanduser().resolve()
    if args.num_datasets <= 0:
        raise ValueError("--num_datasets must be positive.")

    print("=" * 100)
    print("Generating fixed-graph 4-node benchmark datasets")
    print(f"num_nonisomorphic_graphs: {len(dags)}")
    print(f"selected_graphs:          {len(selected_graphs)}")
    print(f"generators:               {generators}")
    print(f"sample_sizes:             {sample_sizes}")
    print(f"num_datasets per file:    {args.num_datasets}")
    print(f"seed:                     {args.seed}")
    print(f"save_root:                {save_root}")
    print("=" * 100)

    for generator_name in generators:
        for graph_info in selected_graphs:
            for num_samples in sample_sizes:
                generator_obj, mechanism_name, noise_name = build_generator(generator_name, args.noise)
                del generator_obj
                output_dir = build_output_dir(
                    save_root=save_root,
                    graph_info=graph_info,
                    generator_name=generator_name,
                    mechanism_name=mechanism_name,
                    noise_name=noise_name,
                    num_samples=num_samples,
                    seed=args.seed,
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                h5_path = output_dir / f"benchmark_4var_numdatasets_{args.num_datasets}.h5"
                config_path = output_dir / "config.json"
                dataset_meta_path = output_dir / "dataset_meta.json"
                print(
                    f"graph_id={int(graph_info['graph_id']):02d} n={num_samples} "
                    f"generator={generator_name} graph={graph_info['name']}"
                )
                generate_h5(
                    h5_path=h5_path,
                    config_path=config_path,
                    dataset_meta_path=dataset_meta_path,
                    graph_info=graph_info,
                    generator_name=generator_name,
                    mechanism_name=mechanism_name,
                    noise_name=noise_name,
                    num_samples=num_samples,
                    num_datasets=args.num_datasets,
                    seed=args.seed,
                    overwrite=args.overwrite,
                )

    print("=" * 100)
    print("Done.")


if __name__ == "__main__":
    main()
