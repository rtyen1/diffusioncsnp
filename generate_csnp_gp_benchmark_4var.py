#!/usr/bin/env python3
"""Generate CSNP-style GP/ER benchmark datasets for 4 variables.

This script mirrors the data distribution used by
ml2_meta_causal_discovery/datasets/create_save_synth_data.py, but it lets us
generate benchmark/test files at multiple observational sample sizes.

Output h5 files contain:
  data:  [num_datasets_in_file, num_samples, num_nodes]
  label: [num_datasets_in_file, num_nodes, num_nodes]

The adjacency convention is label[i, j] = 1 meaning i -> j.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
import tensorflow as tf

from ml2_meta_causal_discovery.datasets.dataset_generators import ClassifyDatasetGenerator


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def json_dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def build_generator(
    *,
    num_nodes: int,
    num_samples: int,
    batch_size: int,
    exp_edges_lower: int,
    exp_edges_upper: int,
) -> ClassifyDatasetGenerator:
    graph_degrees_lower = exp_edges_lower * num_nodes
    graph_degrees_upper = exp_edges_upper * num_nodes
    return ClassifyDatasetGenerator(
        num_variables=num_nodes,
        function_generator="gp",
        batch_size=batch_size,
        num_samples=num_samples,
        kernel_sum=True,
        mean_function="latent",
        graph_type=["ER"],
        graph_degrees=list(range(graph_degrees_lower, graph_degrees_upper + 1)),
    )


def make_output_dir(
    *,
    save_root: Path,
    num_nodes: int,
    num_samples: int,
    seed: int,
    exp_edges_lower: int,
    exp_edges_upper: int,
) -> Path:
    dataset_name = f"csnp_gp_{num_nodes}var_ERL{exp_edges_lower}U{exp_edges_upper}"
    return save_root / dataset_name / f"n_{num_samples}" / f"seed_{seed}"


def write_config(
    *,
    output_dir: Path,
    save_root: Path,
    num_nodes: int,
    num_samples: int,
    total_num_datasets: int,
    datasets_per_file: int,
    seed: int,
    exp_edges_lower: int,
    exp_edges_upper: int,
    num_parts: int,
    overwrite: bool,
) -> None:
    graph_degrees_lower = exp_edges_lower * num_nodes
    graph_degrees_upper = exp_edges_upper * num_nodes
    config: Dict[str, Any] = {
        "distribution": "csnp_gp_er",
        "save_root": str(save_root),
        "n_nodes": int(num_nodes),
        "num_samples": int(num_samples),
        "num_datasets": int(total_num_datasets),
        "datasets_per_file": int(datasets_per_file),
        "num_parts": int(num_parts),
        "seed": int(seed),
        "function_generator": "gp",
        "graph_type": ["ER"],
        "exp_edges_lower_arg": int(exp_edges_lower),
        "exp_edges_upper_arg": int(exp_edges_upper),
        "graph_degrees_lower": int(graph_degrees_lower),
        "graph_degrees_upper": int(graph_degrees_upper),
        "observational_only": True,
        "datasets_in_h5": {
            "data": ["num_datasets_in_file", num_samples, num_nodes],
            "label": ["num_datasets_in_file", num_nodes, num_nodes],
        },
        "notes": {
            "adjacency": "label[i, j] = 1 means i -> j",
            "data_distribution": (
                "Generated with ClassifyDatasetGenerator using GPFunctionGenerator, "
                "ER DAGs, and the same graph-degree convention as create_save_synth_data.py."
            ),
        },
    }
    config_path = output_dir / "config.json"
    if config_path.exists() and not overwrite:
        return
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def generate_one_part(
    *,
    output_dir: Path,
    num_nodes: int,
    num_samples: int,
    part_idx: int,
    part_num_datasets: int,
    part_seed: int,
    exp_edges_lower: int,
    exp_edges_upper: int,
    overwrite: bool,
) -> Path:
    h5_path = output_dir / f"benchmark_{num_nodes}var_csnp_gp_numdatasets_{part_num_datasets}_part_{part_idx:03d}.h5"
    if h5_path.exists() and not overwrite:
        print(f"[SKIP] exists: {h5_path}")
        return h5_path

    np.random.seed(part_seed)
    tf.random.set_seed(part_seed)
    generator = build_generator(
        num_nodes=num_nodes,
        num_samples=num_samples,
        batch_size=part_num_datasets,
        exp_edges_lower=exp_edges_lower,
        exp_edges_upper=exp_edges_upper,
    )
    data, labels = next(generator.generate_next_dataset())
    data = np.asarray(data, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int8)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("data", data=data, compression="gzip")
        f.create_dataset("label", data=labels, compression="gzip")
        f.attrs["distribution"] = "csnp_gp_er"
        f.attrs["n_nodes"] = int(num_nodes)
        f.attrs["num_samples"] = int(num_samples)
        f.attrs["num_datasets"] = int(part_num_datasets)
        f.attrs["part_idx"] = int(part_idx)
        f.attrs["part_seed"] = int(part_seed)
        f.attrs["function_generator"] = "gp"
        f.attrs["graph_type"] = json_dumps(["ER"])
        f.attrs["exp_edges_lower_arg"] = int(exp_edges_lower)
        f.attrs["exp_edges_upper_arg"] = int(exp_edges_upper)
        f.attrs["graph_degrees_lower"] = int(exp_edges_lower * num_nodes)
        f.attrs["graph_degrees_upper"] = int(exp_edges_upper * num_nodes)
        f.attrs["label_convention"] = "label[i, j] = 1 means i -> j"

    print(f"[WRITE] {h5_path} data={tuple(data.shape)} label={tuple(labels.shape)}")
    return h5_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 4-variable CSNP-style GP/ER benchmark test datasets."
    )
    parser.add_argument("--num_nodes", type=int, default=4)
    parser.add_argument("--sample_sizes", type=str, default="5,20,50,100,300,1000,3000")
    parser.add_argument("--num_datasets", type=int, default=1000)
    parser.add_argument("--datasets_per_file", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_edges_lower", type=int, default=0)
    parser.add_argument("--exp_edges_upper", type=int, default=1)
    parser.add_argument(
        "--save_root",
        type=str,
        default="benchmark_data_4var",
        help="Root directory for benchmark data.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_nodes <= 0:
        raise ValueError("--num_nodes must be positive.")
    if args.num_datasets <= 0:
        raise ValueError("--num_datasets must be positive.")
    if args.datasets_per_file <= 0:
        raise ValueError("--datasets_per_file must be positive.")

    sample_sizes = parse_int_list(args.sample_sizes)
    if not sample_sizes:
        raise ValueError("--sample_sizes must contain at least one integer.")

    save_root = Path(args.save_root).expanduser().resolve()
    master_rng = np.random.default_rng(args.seed)

    print("=" * 100)
    print("Generating CSNP-style GP/ER benchmark datasets")
    print(f"num_nodes:          {args.num_nodes}")
    print(f"sample_sizes:       {sample_sizes}")
    print(f"num_datasets:       {args.num_datasets}")
    print(f"datasets_per_file:  {args.datasets_per_file}")
    print(f"seed:               {args.seed}")
    print(f"save_root:          {save_root}")
    print("=" * 100)

    for num_samples in sample_sizes:
        if num_samples <= 0:
            raise ValueError(f"Sample size must be positive, got {num_samples}.")

        output_dir = make_output_dir(
            save_root=save_root,
            num_nodes=args.num_nodes,
            num_samples=num_samples,
            seed=args.seed,
            exp_edges_lower=args.exp_edges_lower,
            exp_edges_upper=args.exp_edges_upper,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        num_parts = int(math.ceil(args.num_datasets / args.datasets_per_file))
        write_config(
            output_dir=output_dir,
            save_root=save_root,
            num_nodes=args.num_nodes,
            num_samples=num_samples,
            total_num_datasets=args.num_datasets,
            datasets_per_file=args.datasets_per_file,
            seed=args.seed,
            exp_edges_lower=args.exp_edges_lower,
            exp_edges_upper=args.exp_edges_upper,
            num_parts=num_parts,
            overwrite=args.overwrite,
        )

        print("-" * 100)
        print(f"num_samples={num_samples} output_dir={output_dir}")

        for part_idx in range(num_parts):
            start = part_idx * args.datasets_per_file
            end = min(args.num_datasets, start + args.datasets_per_file)
            part_num_datasets = end - start
            part_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
            generate_one_part(
                output_dir=output_dir,
                num_nodes=args.num_nodes,
                num_samples=num_samples,
                part_idx=part_idx,
                part_num_datasets=part_num_datasets,
                part_seed=part_seed,
                exp_edges_lower=args.exp_edges_lower,
                exp_edges_upper=args.exp_edges_upper,
                overwrite=args.overwrite,
            )

    print("=" * 100)
    print("Done.")


if __name__ == "__main__":
    main()
