from pathlib import Path
import argparse
import json
import h5py
import numpy as np
from tqdm import tqdm


def sample_one_dataset(num_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate one observational dataset with fixed DAG X -> Y -> Z.
    Variables are ordered as [X, Y, Z].

    Structural equations:
        X = noise_x
        Y = f(X) + eps_y
        Z = g(Y) + eps_z

    We randomly resample the nonlinear mechanisms f and g for every dataset,
    but the graph structure is always fixed.
    """
    # Root node
    x = rng.normal(loc=0.0, scale=rng.uniform(0.8, 1.5), size=num_samples)

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

    noise_scale_y = rng.uniform(0.2, 0.8)
    eps_y = rng.normal(loc=0.0, scale=noise_scale_y, size=num_samples)
    y = fx + eps_y

    # Random nonlinear mechanism for Y -> Z
    ay = rng.uniform(0.8, 1.8)
    by = rng.uniform(0.5, 1.8)
    cy = rng.uniform(-0.6, 0.6)
    dy = rng.uniform(-0.3, 0.3)
    gy_type = rng.integers(0, 3)
    if gy_type == 0:
        gy = ay * np.tanh(by * y + cy) + dy * y
    elif gy_type == 1:
        gy = ay * np.sin(by * y + cy) + dy * (y ** 2)
    else:
        gy = ay * (y / (1.0 + np.abs(by * y))) + dy * np.cos(y + cy)

    noise_scale_z = rng.uniform(0.2, 0.8)
    eps_z = rng.normal(loc=0.0, scale=noise_scale_z, size=num_samples)
    z = gy + eps_z

    data = np.stack([x, y, z], axis=1).astype(np.float32)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="manual_3var_x_to_y_to_z")
    parser.add_argument("--folder_name", type=str, default="test")
    parser.add_argument("--data_start", type=int, default=0)
    parser.add_argument("--data_end", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)

    # Fixed DAG: X -> Y -> Z
    fixed_dag = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
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
        "graph_type": ["FIXED_X_TO_Y_TO_Z"],
        "graph_degrees_upper": 1,
        "graph_degrees_lower": 1,
        "num_variables": 3,
        "num_samples": args.num_samples,
        "function_generator": "manual_chain_nonlinear",
        "description": "All graphs are fixed as X->Y->Z; data are generated manually without GPFunctionGenerator.",
    }
    with open(save_folder / "graph_args.json", "w") as f:
        json.dump(graph_args, f, indent=2)

    print("Saved dataset to:", save_folder)
    print("Example label:")
    print(fixed_dag)


if __name__ == "__main__":
    main()
