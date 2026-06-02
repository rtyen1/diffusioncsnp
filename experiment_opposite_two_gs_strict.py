#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict empirical search for the best free D x D Gumbel-matching distribution
on the opposite-two-modes target:

    P_target(0123) = 0.5
    P_target(3210) = 0.5

The goal is not to train a neural network. The goal is to optimize the free
matrix alpha[node, position] as hard as is practical, then independently
validate whether the induced Gumbel-matching distribution can fit this target.

For D=4, all 24 permutations are enumerated. For each Gumbel noise sample, the
sampled permutation is

    argmax_perm sum_pos alpha[node_at_pos[pos], pos]
                     + noise[node_at_pos[pos], pos].

This is the hard assignment distribution behind hard Gumbel-Sinkhorn, written
directly by enumeration for transparency.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


Perm = Tuple[int, ...]


def perm_to_str(perm: Perm) -> str:
    return "".join(str(x) for x in perm)


def all_perms(num_nodes: int) -> List[Perm]:
    return list(itertools.permutations(range(num_nodes)))


def make_opposite_two_target(perms: Sequence[Perm], num_nodes: int) -> np.ndarray:
    target = np.zeros(len(perms), dtype=np.float64)
    perm_to_idx = {perm: idx for idx, perm in enumerate(perms)}
    target[perm_to_idx[tuple(range(num_nodes))]] = 0.5
    target[perm_to_idx[tuple(reversed(range(num_nodes)))]] = 0.5
    return target


def sample_gumbel(rng: np.random.Generator, shape: Tuple[int, ...]) -> np.ndarray:
    u = rng.uniform(low=np.finfo(np.float32).tiny, high=1.0, size=shape).astype(np.float32)
    return (-np.log(-np.log(u))).astype(np.float32)


def precompute_noise_perm_scores(noise: np.ndarray, perms: np.ndarray) -> np.ndarray:
    scores = np.empty((noise.shape[0], len(perms)), dtype=np.float32)
    positions = np.arange(perms.shape[1])
    for idx, perm in enumerate(perms):
        scores[:, idx] = noise[:, perm, positions].sum(axis=1)
    return scores


def normalize_alpha(theta: np.ndarray, num_nodes: int) -> np.ndarray:
    alpha = theta.reshape(num_nodes, num_nodes).astype(np.float32).copy()
    # Row and column shifts add constants to all assignments, so they do not
    # change the induced permutation distribution. Center them away to remove
    # redundant drift during optimization.
    alpha -= alpha.mean(axis=1, keepdims=True)
    alpha -= alpha.mean(axis=0, keepdims=True)
    alpha -= alpha.mean()
    return alpha


def alpha_perm_scores(alpha: np.ndarray, perms: np.ndarray) -> np.ndarray:
    positions = np.arange(perms.shape[1])
    return alpha[perms, positions].sum(axis=1).astype(np.float32)


def probs_from_scores(alpha: np.ndarray, perms: np.ndarray, noise_perm_scores: np.ndarray) -> np.ndarray:
    scores = noise_perm_scores + alpha_perm_scores(alpha, perms)[None, :]
    winners = scores.argmax(axis=1)
    counts = np.bincount(winners, minlength=len(perms)).astype(np.float64)
    return counts / counts.sum()


def counts_with_fresh_noise(
    alpha: np.ndarray,
    perms: np.ndarray,
    rng: np.random.Generator,
    num_samples: int,
    chunk_size: int,
) -> np.ndarray:
    counts = np.zeros(len(perms), dtype=np.int64)
    remaining = num_samples
    while remaining > 0:
        n = min(chunk_size, remaining)
        noise = sample_gumbel(rng, (n, alpha.shape[0], alpha.shape[1]))
        noise_scores = precompute_noise_perm_scores(noise, perms)
        scores = noise_scores + alpha_perm_scores(alpha, perms)[None, :]
        winners = scores.argmax(axis=1)
        counts += np.bincount(winners, minlength=len(perms))
        remaining -= n
    return counts


def tv_distance(target: np.ndarray, probs: np.ndarray) -> float:
    return float(0.5 * np.abs(target - probs).sum())


def kl_target_to_model(target: np.ndarray, probs: np.ndarray, eps: float = 1e-12) -> float:
    mask = target > 0
    return float((target[mask] * (np.log(target[mask] + eps) - np.log(probs[mask] + eps))).sum())


def target_support_mass(target: np.ndarray, probs: np.ndarray) -> float:
    return float(probs[target > 0].sum())


def entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    mask = probs > 0
    return float(-(probs[mask] * np.log(probs[mask] + eps)).sum())


def metric_row(prefix: str, target: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_tv": tv_distance(target, probs),
        f"{prefix}_kl_target_to_model": kl_target_to_model(target, probs),
        f"{prefix}_target_support_mass": target_support_mass(target, probs),
        f"{prefix}_entropy": entropy(probs),
    }


def bootstrap_metric_ci(
    *,
    counts: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    num_bootstrap: int,
) -> Dict[str, float]:
    total = int(counts.sum())
    empirical = counts / total
    tv_vals = np.empty(num_bootstrap, dtype=np.float64)
    support_vals = np.empty(num_bootstrap, dtype=np.float64)
    for b in range(num_bootstrap):
        boot_counts = rng.multinomial(total, empirical)
        boot_probs = boot_counts / total
        tv_vals[b] = tv_distance(target, boot_probs)
        support_vals[b] = target_support_mass(target, boot_probs)
    return {
        "test_tv_ci_low": float(np.quantile(tv_vals, 0.025)),
        "test_tv_ci_high": float(np.quantile(tv_vals, 0.975)),
        "test_support_ci_low": float(np.quantile(support_vals, 0.025)),
        "test_support_ci_high": float(np.quantile(support_vals, 0.975)),
    }


def make_objective(target: np.ndarray, perms: np.ndarray, train_noise_scores: np.ndarray, num_nodes: int):
    def objective(theta: np.ndarray) -> float:
        alpha = normalize_alpha(theta, num_nodes)
        probs = probs_from_scores(alpha, perms, train_noise_scores)
        return tv_distance(target, probs)

    return objective


def random_search_topk(
    *,
    rng: np.random.Generator,
    objective,
    dim: int,
    bound: float,
    num_trials: int,
    top_k: int,
) -> List[Tuple[float, np.ndarray]]:
    best: List[Tuple[float, np.ndarray]] = []
    for idx in range(num_trials):
        theta = rng.uniform(-bound, bound, size=dim).astype(np.float64)
        loss = float(objective(theta))
        best.append((loss, theta.copy()))
        if len(best) > top_k:
            best.sort(key=lambda x: x[0])
            best = best[:top_k]
        if (idx + 1) % max(1, num_trials // 10) == 0:
            print(f"  random {idx + 1}/{num_trials}: best train TV={best[0][0]:.4f}")
    best.sort(key=lambda x: x[0])
    return best


def local_refine_many(
    *,
    objective,
    candidates: Sequence[Tuple[float, np.ndarray]],
    bounds: Sequence[Tuple[float, float]],
    maxiter: int,
    xtol: float,
    ftol: float,
) -> List[Tuple[float, np.ndarray, str]]:
    from scipy.optimize import minimize

    refined: List[Tuple[float, np.ndarray, str]] = []
    for idx, (loss, theta) in enumerate(candidates):
        print(f"  local Powell from candidate {idx + 1}/{len(candidates)} start TV={loss:.4f}")
        result = minimize(
            objective,
            theta,
            method="Powell",
            bounds=bounds,
            options={"maxiter": maxiter, "xtol": xtol, "ftol": ftol},
        )
        final_loss = float(result.fun)
        final_theta = result.x.copy()
        refined.append((final_loss, final_theta, f"powell_from_candidate_{idx}"))
        print(f"    final train TV={final_loss:.4f}, success={result.success}")
    refined.sort(key=lambda x: x[0])
    return refined


def run_search(args: argparse.Namespace) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    from scipy.optimize import differential_evolution, dual_annealing

    rng = np.random.default_rng(args.seed)
    perms_list = all_perms(args.num_nodes)
    perms = np.asarray(perms_list, dtype=np.int64)
    target = make_opposite_two_target(perms_list, args.num_nodes)
    dim = args.num_nodes * args.num_nodes
    bounds = [(-args.bound, args.bound)] * dim

    print("=" * 100)
    print("Strict opposite-two search for free D x D Gumbel matching")
    print(f"num_nodes:        {args.num_nodes}")
    print(f"num_perms:        {len(perms_list)}")
    print(f"target modes:     {perm_to_str(tuple(range(args.num_nodes)))}, {perm_to_str(tuple(reversed(range(args.num_nodes))))}")
    print(f"seed:             {args.seed}")
    print(f"train_noise:      {args.num_train_noise}")
    print(f"test_noise:       {args.num_test_noise}")
    print(f"bound:            [-{args.bound}, {args.bound}]")
    print("=" * 100)

    train_noise = sample_gumbel(rng, (args.num_train_noise, args.num_nodes, args.num_nodes))
    train_noise_scores = precompute_noise_perm_scores(train_noise, perms)
    objective = make_objective(target, perms, train_noise_scores, args.num_nodes)

    candidate_rows: List[Dict[str, object]] = []
    candidates: List[Tuple[float, np.ndarray, str]] = []

    print("[1/4] Random search")
    random_candidates = random_search_topk(
        rng=rng,
        objective=objective,
        dim=dim,
        bound=args.bound,
        num_trials=args.random_trials,
        top_k=args.keep_top_random,
    )
    for rank, (loss, theta) in enumerate(random_candidates):
        candidates.append((loss, theta, f"random_top_{rank}"))

    print("[2/4] Differential evolution restarts")
    for restart in range(args.de_restarts):
        x0 = random_candidates[restart % len(random_candidates)][1]
        de_seed = args.seed + 1000 + restart
        print(f"  DE restart {restart + 1}/{args.de_restarts}, seed={de_seed}")
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=args.de_maxiter,
            popsize=args.de_popsize,
            seed=de_seed,
            polish=False,
            updating="immediate",
            workers=1,
            init="latinhypercube",
            tol=args.de_tol,
            x0=x0,
        )
        loss = float(result.fun)
        candidates.append((loss, result.x.copy(), f"differential_evolution_{restart}"))
        print(f"    train TV={loss:.4f}, nfev={result.nfev}")

    if args.dual_annealing_restarts > 0:
        print("[3/4] Dual annealing restarts")
        for restart in range(args.dual_annealing_restarts):
            da_seed = args.seed + 2000 + restart
            print(f"  dual_annealing restart {restart + 1}/{args.dual_annealing_restarts}, seed={da_seed}")
            result = dual_annealing(
                objective,
                bounds=bounds,
                seed=da_seed,
                maxiter=args.dual_annealing_maxiter,
                no_local_search=True,
                x0=random_candidates[restart % len(random_candidates)][1],
            )
            loss = float(result.fun)
            candidates.append((loss, result.x.copy(), f"dual_annealing_{restart}"))
            print(f"    train TV={loss:.4f}, nfev={result.nfev}")
    else:
        print("[3/4] Dual annealing skipped")

    candidates.sort(key=lambda x: x[0])
    for rank, (loss, theta, source) in enumerate(candidates):
        candidate_rows.append({"rank_before_local": rank, "source": source, "train_tv": loss})

    print("[4/4] Powell local refinement")
    local_inputs = [(loss, theta) for loss, theta, _ in candidates[: args.local_top_k]]
    refined = local_refine_many(
        objective=objective,
        candidates=local_inputs,
        bounds=bounds,
        maxiter=args.local_maxiter,
        xtol=args.local_xtol,
        ftol=args.local_ftol,
    )

    all_final = candidates + refined
    all_final.sort(key=lambda x: x[0])
    best_train_loss, best_theta, best_source = all_final[0]
    best_alpha = normalize_alpha(best_theta, args.num_nodes)
    train_probs = probs_from_scores(best_alpha, perms, train_noise_scores)

    print("=" * 100)
    print(f"Best train TV={best_train_loss:.4f} from {best_source}")
    print("Validating on fresh independent Gumbel noise...")
    test_rng = np.random.default_rng(args.seed + 10_000_000)
    test_counts = counts_with_fresh_noise(
        alpha=best_alpha,
        perms=perms,
        rng=test_rng,
        num_samples=args.num_test_noise,
        chunk_size=args.test_chunk_size,
    )
    test_probs = test_counts / test_counts.sum()
    ci = bootstrap_metric_ci(
        counts=test_counts,
        target=target,
        rng=np.random.default_rng(args.seed + 20_000_000),
        num_bootstrap=args.num_bootstrap,
    )

    summary: Dict[str, object] = {
        "target": "opposite_two",
        "seed": args.seed,
        "num_nodes": args.num_nodes,
        "num_perms": len(perms_list),
        "num_train_noise": args.num_train_noise,
        "num_test_noise": args.num_test_noise,
        "bound": args.bound,
        "random_trials": args.random_trials,
        "de_restarts": args.de_restarts,
        "de_maxiter": args.de_maxiter,
        "de_popsize": args.de_popsize,
        "dual_annealing_restarts": args.dual_annealing_restarts,
        "dual_annealing_maxiter": args.dual_annealing_maxiter,
        "local_top_k": args.local_top_k,
        "local_maxiter": args.local_maxiter,
        "best_source": best_source,
        "best_train_tv_objective": best_train_loss,
        "alpha_json": json.dumps(best_alpha.tolist()),
    }
    summary.update(metric_row("train", target, train_probs))
    summary.update(metric_row("test", target, test_probs))
    summary.update(ci)

    dist_rows = []
    for idx, perm in enumerate(perms_list):
        dist_rows.append(
            {
                "perm": perm_to_str(perm),
                "p_target": float(target[idx]),
                "p_model_train": float(train_probs[idx]),
                "p_model_test": float(test_probs[idx]),
                "test_count": int(test_counts[idx]),
            }
        )

    alpha_rows = []
    for i in range(best_alpha.shape[0]):
        for j in range(best_alpha.shape[1]):
            alpha_rows.append({"node": i, "position": j, "alpha": float(best_alpha[i, j])})

    print(
        f"TEST TV={summary['test_tv']:.4f} "
        f"[{summary['test_tv_ci_low']:.4f}, {summary['test_tv_ci_high']:.4f}], "
        f"support={summary['test_target_support_mass']:.4f} "
        f"[{summary['test_support_ci_low']:.4f}, {summary['test_support_ci_high']:.4f}], "
        f"KL={summary['test_kl_target_to_model']:.4f}"
    )
    print("Top model probabilities:")
    dist_df_tmp = pd.DataFrame(dist_rows).sort_values("p_model_test", ascending=False)
    print(dist_df_tmp.head(args.top_k_print).to_string(index=False))

    return summary, pd.DataFrame(dist_rows), pd.DataFrame(alpha_rows), pd.DataFrame(candidate_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict opposite-two expressivity search.")
    parser.add_argument("--num_nodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_train_noise", type=int, default=200000)
    parser.add_argument("--num_test_noise", type=int, default=1000000)
    parser.add_argument("--test_chunk_size", type=int, default=100000)
    parser.add_argument("--num_bootstrap", type=int, default=1000)
    parser.add_argument("--bound", type=float, default=20.0)
    parser.add_argument("--random_trials", type=int, default=5000)
    parser.add_argument("--keep_top_random", type=int, default=20)
    parser.add_argument("--de_restarts", type=int, default=5)
    parser.add_argument("--de_maxiter", type=int, default=120)
    parser.add_argument("--de_popsize", type=int, default=12)
    parser.add_argument("--de_tol", type=float, default=1e-4)
    parser.add_argument("--dual_annealing_restarts", type=int, default=2)
    parser.add_argument("--dual_annealing_maxiter", type=int, default=250)
    parser.add_argument("--local_top_k", type=int, default=10)
    parser.add_argument("--local_maxiter", type=int, default=300)
    parser.add_argument("--local_xtol", type=float, default=1e-4)
    parser.add_argument("--local_ftol", type=float, default=1e-4)
    parser.add_argument("--results_dir", type=str, default="all_experiments/gs_opposite_two_strict_results")
    parser.add_argument("--top_k_print", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_nodes != 4:
        raise ValueError("This strict script is intended for num_nodes=4 opposite modes.")

    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    summary, dist_df, alpha_df, candidates_df = run_search(args)

    summary_path = results_dir / "summary_opposite_two_strict.csv"
    dist_path = results_dir / "distribution_opposite_two_strict.csv"
    alpha_path = results_dir / "best_alpha_opposite_two_strict.csv"
    candidates_path = results_dir / "candidate_trace_opposite_two_strict.csv"

    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    dist_df.to_csv(dist_path, index=False)
    alpha_df.to_csv(alpha_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)

    print("=" * 100)
    print(f"Finished in {time.time() - start:.1f}s")
    print(f"Wrote summary:       {summary_path}")
    print(f"Wrote distribution:  {dist_path}")
    print(f"Wrote best alpha:    {alpha_path}")
    print(f"Wrote candidates:    {candidates_path}")


if __name__ == "__main__":
    main()
