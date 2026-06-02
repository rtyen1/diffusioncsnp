from pathlib import Path
import argparse
import json
import h5py
import numpy as np
from tqdm import tqdm

from ml2_meta_causal_discovery.datasets.functions_generator import GPFunctionGenerator


def build_fixed_dag_20():
    """
    Build a 20-node DAG with exactly two edges:
        1 -> 3
        2 -> 3
    in human-friendly 1-based indexing.

    In Python 0-based indexing this means:
        0 -> 2
        1 -> 2
    """
    dag = np.zeros((20, 20), dtype=np.float32)
    dag[0, 2] = 1.0   # 1 -> 3
    dag[1, 2] = 1.0   # 2 -> 3
    return dag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="gp_20var_fixed_1_2_to_3")
    parser.add_argument("--folder_name", type=str, default="test")
    parser.add_argument("--data_start", type=int, default=0)
    parser.add_argument("--data_end", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)

    # Fixed DAG: 1 -> 3, 2 -> 3, all others 0
    fixed_dag = build_fixed_dag_20()

    # IMPORTANT:
    # Use the original GP generator so the sampling logic stays consistent
    # with create_save_synth_data.py when function_generator="gp".
    generator = GPFunctionGenerator(
        num_variables=20,
        num_samples=args.num_samples,
        interventions=False,
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

    print("Saving to:", save_folder)
    print("Fixed DAG edge count:", int(fixed_dag.sum()))
    print("Fixed DAG adjacency matrix:")
    print(fixed_dag.astype(int))

    for seed in tqdm(range(args.data_start, args.data_end), desc="Generating files"):
        np.random.seed(seed)

        all_data = np.zeros((args.batch_size, args.num_samples, 20), dtype=np.float32)
        all_graphs = np.zeros((args.batch_size, 20, 20), dtype=np.float32)

        for b in range(args.batch_size):
            # This is the key line:
            # same graph, but each dataset gets a freshly sampled GP-SCM,
            # then 1000 samples are drawn from that SCM.
            data = generator.generate_data(
                causal_graph=fixed_dag,
                num_int_samples=args.num_samples,
            )

            all_data[b] = data.astype(np.float32)
            all_graphs[b] = fixed_dag

        out_file = save_folder / f"{args.dataset_name}_{seed}.hdf5"
        with h5py.File(out_file, "w") as f:
            f.create_dataset("data", data=all_data)
            f.create_dataset("label", data=all_graphs)

        print(f"Saved: {out_file}")

    graph_args = {
        "graph_type": ["FIXED_1_2_TO_3"],
        "graph_degrees_upper": 2,
        "graph_degrees_lower": 2,
        "num_variables": 20,
        "num_samples": args.num_samples,
        "function_generator": "gp",
        "description": "All graphs are fixed as 1->3 and 2->3 on 20 nodes; all other edges are absent."
    }

    with open(save_folder / "graph_args.json", "w") as f:
        json.dump(graph_args, f, indent=2)

    print("Saved graph_args.json")


if __name__ == "__main__":
    main()