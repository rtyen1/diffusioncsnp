from pathlib import Path
import argparse
import json
import h5py
import numpy as np
from tqdm import tqdm


def sample_one_dataset(num_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate one observational dataset with fixed DAG X -> Y <- Z.
    Variables are ordered as [X, Y, Z].

    Structural equations:
        X = noise_x
        Z = noise_z
        Y = f(X) + g(Z) + eps_y

    We randomly resample the nonlinear mechanisms f and g for every dataset,
    but the graph structure is always fixed.
    """
    # Independent exogenous noises for root nodes
    x = rng.normal(loc=0.0, scale=rng.uniform(0.8, 1.5), size=num_samples)
    z = rng.normal(loc=0.0, scale=rng.uniform(0.8, 1.5), size=num_samples)

    # Random nonlinear mechanism for X -> Y
    ax = rng.uniform(0.8, 1.8)
    bx = rng.uniform(0.5, 1.8)
    cx = rng.uniform(-0.6, 0.6)
    dx = rng.uniform(-0.3, 0.3)
    fx_type = rng.integers(0, 3)
    if fx_type == 0:
        fx = ax * np.tanh(bx * x + cx) + dx * x
    elif fx_type == 1:
        fx = ax * np.sin(bx * x + cx) + dx * (x ** 2)
    else:
        fx = ax * (x / (1.0 + np.abs(bx * x))) + dx * np.cos(x + cx)

    # Random nonlinear mechanism for Z -> Y
    az = rng.uniform(0.8, 1.8)
    bz = rng.uniform(0.5, 1.8)
    cz = rng.uniform(-0.6, 0.6)
    dz = rng.uniform(-0.3, 0.3)
    fz_type = rng.integers(0, 3)
    if fz_type == 0:
        fz = az * np.tanh(bz * z + cz) + dz * z
    elif fz_type == 1:
        fz = az * np.sin(bz * z + cz) + dz * (z ** 2)
    else:
        fz = az * (z / (1.0 + np.abs(bz * z))) + dz * np.cos(z + cz)

    # Child noise: independent additive noise
    noise_scale_y = rng.uniform(0.2, 0.8)
    eps_y = rng.normal(loc=0.0, scale=noise_scale_y, size=num_samples)

    # Fixed graph: X -> Y and Z -> Y only
    y = fx + fz + eps_y

    data = np.stack([x, y, z], axis=1).astype(np.float32)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="manual_3var_xzy_to_y")
    parser.add_argument("--folder_name", type=str, default="test")
    parser.add_argument("--data_start", type=int, default=0)
    parser.add_argument("--data_end", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)

    # Fixed DAG: X -> Y, Z -> Y
    fixed_dag = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )

    save_folder = (
        work_dir
        / "ml2_meta_causal_discovery"
        / "datasets"
        / "data"
        / "synth_training_data"
        / args.dataset_name
        / args.folder_name
    )
    save_folder.mkdir(parents=True, exist_ok=True)

    for seed in tqdm(range(args.data_start, args.data_end)):
        rng = np.random.default_rng(seed)

        all_data = np.zeros((args.batch_size, args.num_samples, 3), dtype=np.float32)
        all_graphs = np.zeros((args.batch_size, 3, 3), dtype=np.float32)

        for b in range(args.batch_size):
            data = sample_one_dataset(num_samples=args.num_samples, rng=rng)
            all_data[b] = data
            all_graphs[b] = fixed_dag

        out_file = save_folder / f"{args.dataset_name}_{seed}.hdf5"
        with h5py.File(out_file, "w") as f:
            f.create_dataset("data", data=all_data)
            f.create_dataset("label", data=all_graphs)

    graph_args = {
        "graph_type": ["FIXED_X_TO_Y_AND_Z_TO_Y"],
        "graph_degrees_upper": 2,
        "graph_degrees_lower": 2,
        "num_variables": 3,
        "num_samples": args.num_samples,
        "function_generator": "manual_additive_nonlinear",
        "description": "All graphs are fixed as X->Y and Z->Y; data are generated manually without GPFunctionGenerator.",
    }
    with open(save_folder / "graph_args.json", "w") as f:
        json.dump(graph_args, f, indent=2)

    print("Saved dataset to:", save_folder)
    print("Example label:")
    print(fixed_dag)


if __name__ == "__main__":
    main()
