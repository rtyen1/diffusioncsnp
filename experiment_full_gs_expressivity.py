#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the expressivity of a single free D x D Gumbel-matching distribution.

This script is intentionally independent of CSNP. It asks:

    Given a target distribution over permutations, can a free matrix log_alpha
    make hard Gumbel matching sample approximately that distribution?

For small D, we enumerate all D! permutations. A permutation is represented as
node_at_position, e.g. "0123" means node 0 at position 0, node 1 at position 1.

For each Gumbel noise sample, hard Gumbel matching selects

    argmax_perm sum_pos log_alpha[node_at_pos[pos], pos]
                     + noise[node_at_pos[pos], pos].

This is the hard assignment step used by Gumbel-Sinkhorn with hard=True, but
enumerated directly for clarity and for exact small-D scoring.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


Perm = Tuple[int, ...]


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def all_perms(num_nodes: int) -> List[Perm]:
    return list(itertools.permutations(range(num_nodes)))


def perm_to_str(perm: Perm) -> str:
    return "".join(str(x) for x in perm)


def parse_perm(s: str, num_nodes: int) -> Perm:
    s = s.strip()
    if "," in s:
        perm = tuple(int(x) for x in s.split(","))
    else:
        perm = tuple(int(ch) for ch in s)
    if len(perm) != num_nodes or sorted(perm) != list(range(num_nodes)):
        raise ValueError(f"Invalid permutation {s!r} for num_nodes={num_nodes}.")
    return perm


def is_valid_topological_order(order: Perm, edges: Sequence[Tuple[int, int]]) -> bool:
    pos = {node: idx for idx, node in enumerate(order)}
    return all(pos[parent] < pos[child] for parent, child in edges)


def make_target_distribution(target_name: str, perms: Sequence[Perm], num_nodes: int) -> np.ndarray:
    p = np.zeros(len(perms), dtype=np.float64)
    perm_to_idx = {perm: i for i, perm in enumerate(perms)}

    if target_name == "single":
        p[perm_to_idx[tuple(range(num_nodes))]] = 1.0
    elif target_name == "close_two":
        first = tuple(range(num_nodes))
        second = list(range(num_nodes))
        second[-1], second[-2] = second[-2], second[-1]
        p[perm_to_idx[first]] = 0.5
        p[perm_to_idx[tuple(second)]] = 0.5
    elif target_name == "opposite_two":
        p[perm_to_idx[tuple(range(num_nodes))]] = 0.5
        p[perm_to_idx[tuple(reversed(range(num_nodes)))]] = 0.5
    elif target_name == "four_modes":
        modes = [
            tuple(range(num_nodes)),
            (1, 0, 2, 3),
            (2, 3, 0, 1),
            tuple(reversed(range(num_nodes))),
        ]
        if num_nodes != 4:
            raise ValueError("four_modes is currently defined only for num_nodes=4.")
        for mode in modes:
            p[perm_to_idx[mode]] = 0.25
    elif target_name == "topo_0to2_1to2":
        if num_nodes < 3:
            raise ValueError("topo_0to2_1to2 requires num_nodes >= 3.")
        valid = [idx for idx, perm in enumerate(perms) if is_valid_topological_order(perm, [(0, 2), (1, 2)])]
        p[valid] = 1.0 / len(valid)
    elif target_name == "cpdag_chain_uniform_orders":
        if num_nodes != 4:
            raise ValueError("cpdag_chain_uniform_orders is defined for num_nodes=4.")
        # CPDAG 0 - 1 - 2 - 3. Compatible orders are exactly those that
        # do not create an unshielded collider at internal nodes 1 or 2.
        valid = []
        for idx, perm in enumerate(perms):
            pos = {node: rank for rank, node in enumerate(perm)}
            collider_at_1 = pos[0] < pos[1] and pos[2] < pos[1]
            collider_at_2 = pos[1] < pos[2] and pos[3] < pos[2]
            if not collider_at_1 and not collider_at_2:
                valid.append(idx)
        p[valid] = 1.0 / len(valid)
    elif target_name == "cpdag_chain_uniform_dags":
        if num_nodes != 4:
            raise ValueError("cpdag_chain_uniform_dags is defined for num_nodes=4.")
        # Same CPDAG 0 - 1 - 2 - 3. First choose one of the 4 equivalent
        # DAGs uniformly; then choose uniformly among that DAG's topological
        # orders. This is different from uniform over the union of orders.
        weights = {
            (0, 1, 2, 3): 1.0 / 4.0,
            (1, 0, 2, 3): 1.0 / 12.0,
            (1, 2, 0, 3): 1.0 / 12.0,
            (1, 2, 3, 0): 1.0 / 12.0,
            (2, 1, 0, 3): 1.0 / 12.0,
            (2, 1, 3, 0): 1.0 / 12.0,
            (2, 3, 1, 0): 1.0 / 12.0,
            (3, 2, 1, 0): 1.0 / 4.0,
        }
        for perm, prob in weights.items():
            p[perm_to_idx[perm]] = prob
    else:
        raise ValueError(f"Unknown target {target_name!r}.")

    if not np.isclose(p.sum(), 1.0):
        raise RuntimeError(f"Target distribution {target_name} does not sum to 1.")
    return p


def sample_gumbel(rng: np.random.Generator, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
    u = rng.uniform(low=np.finfo(np.float32).tiny, high=1.0, size=shape).astype(dtype)
    return (-np.log(-np.log(u))).astype(dtype)


def precompute_noise_perm_scores(noise: np.ndarray, perms: np.ndarray) -> np.ndarray:
    """Return scores from noise only, shape [num_noise, num_perms]."""
    num_noise = noise.shape[0]
    scores = np.empty((num_noise, len(perms)), dtype=np.float32)
    positions = np.arange(perms.shape[1])
    for idx, perm in enumerate(perms):
        scores[:, idx] = noise[:, perm, positions].sum(axis=1)
    return scores


def alpha_perm_scores(alpha: np.ndarray, perms: np.ndarray) -> np.ndarray:
    positions = np.arange(perms.shape[1])
    return alpha[perms, positions].sum(axis=1).astype(np.float32)


def probs_from_precomputed_scores(
    alpha: np.ndarray,
    perms: np.ndarray,
    noise_perm_scores: np.ndarray,
) -> np.ndarray:
    scores = noise_perm_scores + alpha_perm_scores(alpha, perms)[None, :]
    winners = scores.argmax(axis=1)
    counts = np.bincount(winners, minlength=len(perms)).astype(np.float64)
    return counts / counts.sum()


def estimate_probs_with_fresh_noise(
    alpha: np.ndarray,
    perms: np.ndarray,
    rng: np.random.Generator,
    num_samples: int,
    chunk_size: int,
) -> np.ndarray:
    counts = np.zeros(len(perms), dtype=np.float64)
    remaining = num_samples
    while remaining > 0:
        n = min(chunk_size, remaining)
        noise = sample_gumbel(rng, (n, alpha.shape[0], alpha.shape[1]))
        noise_scores = precompute_noise_perm_scores(noise, perms)
        scores = noise_scores + alpha_perm_scores(alpha, perms)[None, :]
        winners = scores.argmax(axis=1)
        counts += np.bincount(winners, minlength=len(perms))
        remaining -= n
    return counts / counts.sum()


def normalize_alpha(theta: np.ndarray, num_nodes: int) -> np.ndarray:
    alpha = theta.reshape(num_nodes, num_nodes).astype(np.float32).copy()
    # Row/column shifts do not change assignment preferences. Centering only
    # removes redundant drift and makes saved matrices easier to read.
    alpha -= alpha.mean(axis=1, keepdims=True)
    alpha -= alpha.mean(axis=0, keepdims=True)
    alpha -= alpha.mean()
    return alpha


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    mask = p > 0
    return float((p[mask] * (np.log(p[mask] + eps) - np.log(q[mask] + eps))).sum())


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    mask = p > 0
    return float(-(p[mask] * np.log(p[mask] + eps)).sum())


def support_mass(target: np.ndarray, model: np.ndarray) -> float:
    return float(model[target > 0].sum())


def metrics(target: np.ndarray, model: np.ndarray) -> Dict[str, float]:
    return {
        "tv": tv_distance(target, model),
        "kl_target_to_model": kl_divergence(target, model),
        "target_support_mass": support_mass(target, model),
        "entropy": entropy(model),
    }


def random_search(
    *,
    rng: np.random.Generator,
    target: np.ndarray,
    perms: np.ndarray,
    noise_scores: np.ndarray,
    num_nodes: int,
    num_trials: int,
    bound: float,
) -> Tuple[np.ndarray, float]:
    best_theta = None
    best_loss = math.inf
    for _ in range(num_trials):
        theta = rng.uniform(-bound, bound, size=num_nodes * num_nodes).astype(np.float32)
        alpha = normalize_alpha(theta, num_nodes)
        probs = probs_from_precomputed_scores(alpha, perms, noise_scores)
        loss = tv_distance(target, probs)
        if loss < best_loss:
            best_loss = loss
            best_theta = theta.copy()
    assert best_theta is not None
    return best_theta, best_loss


def optimize_target(
    *,
    target_name: str,
    target: np.ndarray,
    perms: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, object]:
    try:
        from scipy.optimize import differential_evolution, minimize
    except ImportError as e:
        raise ImportError("This script needs scipy.optimize. Install scipy in the csnp environment.") from e

    rng = np.random.default_rng(seed)
    train_noise = sample_gumbel(rng, (args.num_train_noise, args.num_nodes, args.num_nodes))
    train_noise_scores = precompute_noise_perm_scores(train_noise, perms)

    print(f"[{target_name} seed={seed}] random search trials={args.random_trials}")
    best_theta, best_loss = random_search(
        rng=rng,
        target=target,
        perms=perms,
        noise_scores=train_noise_scores,
        num_nodes=args.num_nodes,
        num_trials=args.random_trials,
        bound=args.bound,
    )
    print(f"[{target_name} seed={seed}] random best train TV={best_loss:.4f}")

    bounds = [(-args.bound, args.bound)] * (args.num_nodes * args.num_nodes)

    def objective(theta: np.ndarray) -> float:
        alpha = normalize_alpha(theta, args.num_nodes)
        probs = probs_from_precomputed_scores(alpha, perms, train_noise_scores)
        return tv_distance(target, probs)

    print(
        f"[{target_name} seed={seed}] differential_evolution "
        f"maxiter={args.de_maxiter} popsize={args.de_popsize}"
    )
    de_result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=args.de_maxiter,
        popsize=args.de_popsize,
        seed=seed,
        polish=False,
        updating="immediate",
        workers=1,
        init="latinhypercube",
        tol=args.de_tol,
        x0=best_theta,
    )
    theta = de_result.x if de_result.fun <= best_loss else best_theta
    print(f"[{target_name} seed={seed}] DE best train TV={min(float(de_result.fun), best_loss):.4f}")

    if not args.no_local:
        print(f"[{target_name} seed={seed}] local Powell maxiter={args.local_maxiter}")
        local_result = minimize(
            objective,
            theta,
            method="Powell",
            bounds=bounds,
            options={"maxiter": args.local_maxiter, "xtol": args.local_xtol, "ftol": args.local_ftol},
        )
        if local_result.fun < objective(theta):
            theta = local_result.x
        print(f"[{target_name} seed={seed}] local best train TV={objective(theta):.4f}")

    alpha = normalize_alpha(theta, args.num_nodes)
    train_probs = probs_from_precomputed_scores(alpha, perms, train_noise_scores)

    test_rng = np.random.default_rng(seed + 10_000_000)
    test_probs = estimate_probs_with_fresh_noise(
        alpha=alpha,
        perms=perms,
        rng=test_rng,
        num_samples=args.num_test_noise,
        chunk_size=args.test_chunk_size,
    )

    return {
        "target_name": target_name,
        "seed": seed,
        "alpha": alpha,
        "train_probs": train_probs,
        "test_probs": test_probs,
        "train_metrics": metrics(target, train_probs),
        "test_metrics": metrics(target, test_probs),
    }


def build_rows(
    *,
    result: Dict[str, object],
    target: np.ndarray,
    perms: Sequence[Perm],
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    target_name = str(result["target_name"])
    seed = int(result["seed"])
    train_metrics = result["train_metrics"]
    test_metrics = result["test_metrics"]
    alpha = np.asarray(result["alpha"])
    train_probs = np.asarray(result["train_probs"])
    test_probs = np.asarray(result["test_probs"])

    summary = {
        "target": target_name,
        "seed": seed,
        "num_nodes": args.num_nodes,
        "num_perms": len(perms),
        "num_train_noise": args.num_train_noise,
        "num_test_noise": args.num_test_noise,
        "bound": args.bound,
        "random_trials": args.random_trials,
        "de_maxiter": args.de_maxiter,
        "de_popsize": args.de_popsize,
        "local_enabled": not args.no_local,
        "train_tv": train_metrics["tv"],
        "train_kl_target_to_model": train_metrics["kl_target_to_model"],
        "train_target_support_mass": train_metrics["target_support_mass"],
        "train_entropy": train_metrics["entropy"],
        "test_tv": test_metrics["tv"],
        "test_kl_target_to_model": test_metrics["kl_target_to_model"],
        "test_target_support_mass": test_metrics["target_support_mass"],
        "test_entropy": test_metrics["entropy"],
        "alpha_json": json.dumps(alpha.tolist()),
    }

    dist_rows = []
    for idx, perm in enumerate(perms):
        dist_rows.append(
            {
                "target": target_name,
                "seed": seed,
                "perm": perm_to_str(perm),
                "p_target": float(target[idx]),
                "p_model_train": float(train_probs[idx]),
                "p_model_test": float(test_probs[idx]),
            }
        )

    alpha_rows = []
    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            alpha_rows.append(
                {
                    "target": target_name,
                    "seed": seed,
                    "node": i,
                    "position": j,
                    "alpha": float(alpha[i, j]),
                }
            )
    return summary, dist_rows, alpha_rows


def print_top_distribution(target_name: str, target: np.ndarray, probs: np.ndarray, perms: Sequence[Perm], top_k: int) -> None:
    order = np.argsort(-np.maximum(target, probs))[:top_k]
    print(f"[{target_name}] top distribution rows:")
    for idx in order:
        print(f"  {perm_to_str(perms[idx])}: target={target[idx]:.4f}, model={probs[idx]:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Free D x D Gumbel-matching expressivity experiment.")
    parser.add_argument("--num_nodes", type=int, default=4)
    parser.add_argument(
        "--targets",
        type=str,
        default="single,close_two,opposite_two,topo_0to2_1to2",
        help=(
            "Comma-separated targets: single, close_two, opposite_two, four_modes, "
            "topo_0to2_1to2, cpdag_chain_uniform_orders, cpdag_chain_uniform_dags."
        ),
    )
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--num_train_noise", type=int, default=100000)
    parser.add_argument("--num_test_noise", type=int, default=500000)
    parser.add_argument("--test_chunk_size", type=int, default=100000)
    parser.add_argument("--bound", type=float, default=20.0)
    parser.add_argument("--random_trials", type=int, default=2000)
    parser.add_argument("--de_maxiter", type=int, default=80)
    parser.add_argument("--de_popsize", type=int, default=10)
    parser.add_argument("--de_tol", type=float, default=1e-4)
    parser.add_argument("--no_local", action="store_true")
    parser.add_argument("--local_maxiter", type=int, default=200)
    parser.add_argument("--local_xtol", type=float, default=1e-4)
    parser.add_argument("--local_ftol", type=float, default=1e-4)
    parser.add_argument("--results_dir", type=str, default="all_experiments/gs_expressivity_results")
    parser.add_argument("--summary_name", type=str, default="summary.csv")
    parser.add_argument("--distribution_name", type=str, default="permutation_distributions.csv")
    parser.add_argument("--alpha_name", type=str, default="best_alpha.csv")
    parser.add_argument("--top_k_print", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_nodes > 5:
        raise ValueError("This enumerates all D! permutations; use num_nodes <= 5 for this script.")

    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    perms_list = all_perms(args.num_nodes)
    perms_array = np.asarray(perms_list, dtype=np.int64)
    targets = parse_csv_list(args.targets)
    seeds = [int(x) for x in parse_csv_list(args.seeds)]

    print("=" * 100)
    print("Free D x D Gumbel-matching expressivity experiment")
    print(f"num_nodes:       {args.num_nodes}")
    print(f"num_perms:       {len(perms_list)}")
    print(f"targets:         {targets}")
    print(f"seeds:           {seeds}")
    print(f"train noise:     {args.num_train_noise}")
    print(f"test noise:      {args.num_test_noise}")
    print(f"results_dir:     {results_dir}")
    print("=" * 100)

    summary_rows: List[Dict[str, object]] = []
    distribution_rows: List[Dict[str, object]] = []
    alpha_rows: List[Dict[str, object]] = []
    start = time.time()

    for target_name in targets:
        target = make_target_distribution(target_name, perms_list, args.num_nodes)
        for seed in seeds:
            result = optimize_target(
                target_name=target_name,
                target=target,
                perms=perms_array,
                args=args,
                seed=seed,
            )
            summary, dist, alpha = build_rows(result=result, target=target, perms=perms_list, args=args)
            summary_rows.append(summary)
            distribution_rows.extend(dist)
            alpha_rows.extend(alpha)
            print(
                f"[{target_name} seed={seed}] TEST "
                f"TV={summary['test_tv']:.4f}, "
                f"support={summary['test_target_support_mass']:.4f}, "
                f"KL={summary['test_kl_target_to_model']:.4f}"
            )
            print_top_distribution(target_name, target, np.asarray(result["test_probs"]), perms_list, args.top_k_print)

    summary_df = pd.DataFrame(summary_rows)
    dist_df = pd.DataFrame(distribution_rows)
    alpha_df = pd.DataFrame(alpha_rows)

    summary_path = results_dir / args.summary_name
    dist_path = results_dir / args.distribution_name
    alpha_path = results_dir / args.alpha_name
    summary_df.to_csv(summary_path, index=False)
    dist_df.to_csv(dist_path, index=False)
    alpha_df.to_csv(alpha_path, index=False)

    print("=" * 100)
    print(f"Finished in {time.time() - start:.1f}s")
    print(f"Wrote summary:      {summary_path}")
    print(f"Wrote distributions:{dist_path}")
    print(f"Wrote best alpha:   {alpha_path}")


if __name__ == "__main__":
    main()
