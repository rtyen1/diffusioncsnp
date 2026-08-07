#!/usr/bin/env python3
"""Diagnose each learned reverse transition of a topology diffusion model."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_topo_order_4var_summary import encode_input, load_model, remove_diag  # noqa: E402
from test_code.evaluate_variable_node_joint_checkpoint_sweep import (  # noqa: E402
    discover_variable_h5_files,
    metadata_sem_name,
    select_observations,
    standardize_full_dataset,
)


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def pairwise_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of unordered item pairs with the same relative order."""
    n = int(pred.numel())
    if n < 2:
        return 1.0
    pred_pos = torch.empty_like(pred)
    target_pos = torch.empty_like(target)
    positions = torch.arange(n, device=pred.device)
    pred_pos[pred] = positions
    target_pos[target] = positions
    i, j = torch.triu_indices(n, n, offset=1, device=pred.device)
    agrees = (pred_pos[i] < pred_pos[j]) == (target_pos[i] < target_pos[j])
    return float(agrees.float().mean().item())


def permutation_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    return {
        "exact_match": float(torch.equal(pred, target)),
        "position_accuracy": float((pred == target).float().mean().item()),
        "pairwise_accuracy": pairwise_accuracy(pred, target),
    }


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def diagnose_dataset(
    model,
    model_module: str,
    data: np.ndarray,
    graph: np.ndarray,
    device: str,
    standardize: bool,
    noise_repeats: int,
    seed: int,
) -> List[Dict[str, float]]:
    inputs = encode_input(data, device=device, standardize=standardize)
    graph_tensor = torch.as_tensor(graph, dtype=inputs.dtype, device=device).unsqueeze(0)

    seed_all(seed)
    priority_ordered = None
    if model_module == "topo_priority_diffusion":
        node_repr, priority_ordered, _, _ = model._encode_priority_ordered_data(
            inputs,
            graph=graph_tensor,
            mask=None,
        )
    else:
        node_repr, _ = model._encode_ordered_data(inputs, graph=graph_tensor, mask=None)

    if priority_ordered is None:
        reverse_model = model.reverse_model
    else:
        def reverse_model(src, time, x_start):
            return model.reverse_model(src, time, x_start, priority_ordered)

    reverse_steps = list(model.diffusion_utils.reverse_steps)
    if len(reverse_steps) < 2:
        raise ValueError(f"At least two reverse steps are required, got {reverse_steps}.")

    identity = torch.arange(node_repr.size(1), device=device)
    accumulators: Dict[tuple[int, int], Dict[str, List[float]]] = {}

    for repeat in range(noise_repeats):
        seed_all(seed + repeat * 104729)
        perm_seq = model.diffusion_utils.q_sample_seq(identity.unsqueeze(0))
        perm_seq = perm_seq[:, reverse_steps, ...]

        for step_idx in range(1, len(reverse_steps)):
            previous_step = int(reverse_steps[step_idx - 1])
            current_step = int(reverse_steps[step_idx])
            perm_previous = perm_seq[0, step_idx - 1, 0]
            perm_current = perm_seq[0, step_idx, 0]
            timestep = torch.tensor([current_step], device=device)

            scores = reverse_model(
                perm_current.view(1, 1, -1),
                timestep,
                node_repr,
            ).squeeze(1)
            log_prob = model.diffusion_utils.p_log_cond_prob(
                scores.float(),
                perm_tm1=perm_previous.unsqueeze(0),
                perm_t=perm_current.unsqueeze(0),
            )
            predicted_previous, _, _ = model.diffusion_utils.p_sample(
                reverse_model,
                perm_current.unsqueeze(0),
                timestep,
                node_repr,
                deterministic=True,
            )
            predicted_previous = predicted_previous[0]

            target_metrics = permutation_metrics(predicted_previous, perm_previous)
            current_clean = permutation_metrics(perm_current, identity)
            predicted_clean = permutation_metrics(predicted_previous, identity)
            true_previous_clean = permutation_metrics(perm_previous, identity)

            values = {
                "nll": float((-log_prob).item()),
                "uniform_nll": float(math.lgamma(int(identity.numel()) + 1)),
                "nll_gain_over_uniform": float(
                    math.lgamma(int(identity.numel()) + 1) + log_prob.item()
                ),
                "target_exact_match": target_metrics["exact_match"],
                "target_position_accuracy": target_metrics["position_accuracy"],
                "target_pairwise_accuracy": target_metrics["pairwise_accuracy"],
                "clean_pairwise_before": current_clean["pairwise_accuracy"],
                "clean_pairwise_after": predicted_clean["pairwise_accuracy"],
                "clean_pairwise_improvement": (
                    predicted_clean["pairwise_accuracy"] - current_clean["pairwise_accuracy"]
                ),
                "true_previous_clean_pairwise": true_previous_clean["pairwise_accuracy"],
            }
            bucket = accumulators.setdefault(
                (current_step, previous_step),
                {name: [] for name in values},
            )
            for name, value in values.items():
                bucket[name].append(value)

    rows: List[Dict[str, float]] = []
    for (current_step, previous_step), metrics in accumulators.items():
        row: Dict[str, float] = {
            "step_from": current_step,
            "step_to": previous_step,
            "transition": f"{current_step}->{previous_step}",
        }
        row.update({name: float(np.mean(values)) for name, values in metrics.items()})
        rows.append(row)
    return rows


def aggregate_rows(frame: pd.DataFrame, group_columns: Iterable[str]) -> pd.DataFrame:
    metric_columns = [
        "nll",
        "uniform_nll",
        "nll_gain_over_uniform",
        "target_exact_match",
        "target_position_accuracy",
        "target_pairwise_accuracy",
        "clean_pairwise_before",
        "clean_pairwise_after",
        "clean_pairwise_improvement",
        "true_previous_clean_pairwise",
    ]
    grouped = frame.groupby(list(group_columns), dropna=False, sort=True)
    result = grouped[metric_columns].mean().add_prefix("mean_").reset_index()
    result["n_ok"] = grouped.size().to_numpy()
    return result


def save_plot(aggregate: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    sample_sizes = sorted(aggregate["sample_size"].unique())
    transitions = list(dict.fromkeys(aggregate["transition"].tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for sample_size in sample_sizes:
        part = aggregate[aggregate["sample_size"] == sample_size].copy()
        part["transition"] = pd.Categorical(part["transition"], transitions, ordered=True)
        part = part.sort_values("transition")
        axes[0].plot(part["transition"].astype(str), part["mean_nll"], marker="o", label=f"n={sample_size}")
        axes[1].plot(
            part["transition"].astype(str),
            part["mean_clean_pairwise_improvement"],
            marker="o",
            label=f"n={sample_size}",
        )
    axes[0].set_title("Reverse-step NLL")
    axes[0].set_ylabel("NLL (lower is better)")
    axes[1].set_title("Progress toward clean order")
    axes[1].set_ylabel("Pairwise accuracy improvement")
    for axis in axes:
        axis.set_xlabel("reverse transition")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> None:
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    models_root = Path(args.models_root).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    node_counts = parse_int_list(args.node_counts)
    sample_sizes = parse_int_list(args.sample_sizes)
    files_by_d = discover_variable_h5_files(benchmark_root, args.split, node_counts)

    model, config, loaded_path = load_model(
        models_root=models_root,
        run_name=args.run_name,
        checkpoint=args.checkpoint,
        source="current",
        device=device,
        bak_path=Path("ml2_meta_causal_discovery/models/causaltransformernp.py.mask_version.bak"),
    )
    model_module = config.get("module")
    supported_modules = {"topo_diffusion", "topo_priority_diffusion"}
    if model_module not in supported_modules:
        raise ValueError(
            "This diagnostic supports order-only topo_diffusion and "
            "topo_priority_diffusion models; "
            f"got module={model_module!r}."
        )
    model.eval()

    print("=" * 100)
    print("Topology diffusion reverse-step diagnostic")
    print(f"loaded:          {loaded_path}")
    print(f"module:          {model_module}")
    if model_module == "topo_priority_diffusion":
        print(f"priority_mode:   {getattr(model, 'topo_priority_mode', 'random')}")
    print(f"benchmark_root:  {benchmark_root}")
    print(f"split:           {args.split}")
    print(f"node_counts:     {node_counts}")
    print(f"sample_sizes:    {sample_sizes}")
    print(f"reverse_steps:   {model.diffusion_utils.reverse_steps}")
    print(f"noise_repeats:   {args.noise_repeats}")
    print(f"device:          {device}")
    print("=" * 100)

    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []
    started = time.time()
    for node_count in node_counts:
        files = files_by_d[node_count]
        if not files:
            raise FileNotFoundError(
                f"No d{node_count}_shard_*.hdf5 files under {benchmark_root / args.split}."
            )
        seen = 0
        for file_index, h5_path in enumerate(files):
            with h5py.File(h5_path, "r") as handle:
                count = int(handle["data"].shape[0])
                for data_index in range(count):
                    if args.max_datasets_per_node is not None and seen >= args.max_datasets_per_node:
                        break
                    raw_data = np.asarray(handle["data"][data_index], dtype=np.float32)
                    graph = remove_diag(np.asarray(handle["label"][data_index], dtype=np.float32))
                    sem_type = metadata_sem_name(handle, data_index)
                    normalized_data = standardize_full_dataset(raw_data, args.standardize)
                    seen += 1
                    for sample_size in sample_sizes:
                        eval_seed = (
                            args.seed
                            + node_count * 1000003
                            + file_index * 10007
                            + data_index * 101
                            + sample_size
                        )
                        try:
                            selected = select_observations(normalized_data, sample_size, eval_seed)
                            step_rows = diagnose_dataset(
                                model=model,
                                model_module=model_module,
                                data=selected,
                                graph=graph,
                                device=device,
                                standardize=args.standardize,
                                noise_repeats=args.noise_repeats,
                                seed=eval_seed,
                            )
                            for row in step_rows:
                                row.update(
                                    {
                                        "split": args.split,
                                        "node_count": node_count,
                                        "sample_size": sample_size,
                                        "sem_type": sem_type,
                                        "file": h5_path.name,
                                        "dataset_index": data_index,
                                    }
                                )
                                rows.append(row)
                        except Exception as exc:
                            error = {
                                "split": args.split,
                                "node_count": node_count,
                                "sample_size": sample_size,
                                "file": h5_path.name,
                                "dataset_index": data_index,
                                "error": repr(exc),
                            }
                            errors.append(error)
                            if args.print_errors:
                                print(f"[ERROR] {error}")
                if args.max_datasets_per_node is not None and seen >= args.max_datasets_per_node:
                    break
        print(f"D={node_count}: evaluated {seen} datasets")

    if not rows:
        raise RuntimeError(f"No successful evaluations; errors={len(errors)}.")

    detailed = pd.DataFrame(rows)
    aggregate = aggregate_rows(
        detailed,
        ["split", "node_count", "sample_size", "step_from", "step_to", "transition"],
    ).sort_values(["node_count", "sample_size", "step_from"], ascending=[True, True, False])
    by_sem = aggregate_rows(
        detailed,
        ["split", "node_count", "sample_size", "sem_type", "step_from", "step_to", "transition"],
    ).sort_values(
        ["node_count", "sample_size", "sem_type", "step_from"],
        ascending=[True, True, True, False],
    )

    prefix = args.summary_prefix
    detailed_path = results_dir / f"{prefix}_detailed.csv"
    aggregate_path = results_dir / f"{prefix}_aggregate.csv"
    by_sem_path = results_dir / f"{prefix}_by_sem.csv"
    errors_path = results_dir / f"{prefix}_errors.csv"
    args_path = results_dir / f"{prefix}_args.json"
    detailed.to_csv(detailed_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    by_sem.to_csv(by_sem_path, index=False)
    pd.DataFrame(errors).to_csv(errors_path, index=False)
    with args_path.open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    save_plot(aggregate, results_dir / f"{prefix}_reverse_steps.png")

    print("=" * 100)
    print(f"Finished in {time.time() - started:.1f}s with {len(errors)} errors")
    print(f"aggregate: {aggregate_path}")
    print(f"by SEM:    {by_sem_path}")
    print(f"detailed:  {detailed_path}")
    print(f"plot:      {results_dir / f'{prefix}_reverse_steps.png'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--node_counts", default="10")
    parser.add_argument("--sample_sizes", default="100,300,1000")
    parser.add_argument("--models_root", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--noise_repeats", type=int, default=4)
    parser.add_argument("--max_datasets_per_node", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--standardize", dest="standardize", action="store_true", default=True)
    parser.add_argument("--no_standardize", dest="standardize", action="store_false")
    parser.add_argument("--print_errors", action="store_true")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--summary_prefix", default="topo_reverse_steps")
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
