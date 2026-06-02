from pathlib import Path
import argparse
import json
import h5py
import numpy as np
from tqdm import tqdm

from ml2_meta_causal_discovery.datasets.functions_generator import GPFunctionGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="gp_2var_only_xy")
    parser.add_argument("--folder_name", type=str, default="test")
    parser.add_argument("--data_start", type=int, default=0)
    parser.add_argument("--data_end", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)

    # 固定 DAG: X -> Y
    fixed_dag = np.array([[0, 1],
                          [0, 0]], dtype=np.float32)

    generator = GPFunctionGenerator(
        num_variables=2,
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

    for seed in tqdm(range(args.data_start, args.data_end)):
        np.random.seed(seed)

        all_data = np.zeros(
            (args.batch_size, args.num_samples, 2),
            dtype=np.float32
        )
        all_graphs = np.zeros(
            (args.batch_size, 2, 2),
            dtype=np.float32
        )

        for b in range(args.batch_size):
            # 根据固定 DAG 采样数据
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

    # 保存说明文件
    graph_args = {
        "graph_type": ["FIXED_X_TO_Y"],
        "graph_degrees_upper": 1,
        "graph_degrees_lower": 1,
        "num_variables": 2,
        "num_samples": args.num_samples,
        "function_generator": "gp",
        "description": "All test graphs are fixed as X->Y",
    }
    with open(save_folder / "graph_args.json", "w") as f:
        json.dump(graph_args, f, indent=2)


if __name__ == "__main__":
    main()