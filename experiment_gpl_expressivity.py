#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expressivity tests for generalized Plackett-Luce (GPL) permutation models.

This script is independent of CSNP and uses exact enumeration for small D.

Two experiments are supported:

1. one_step
   Fit a single free GPL score matrix S[D, D] to a target distribution over
   permutations.

2. multistep
   Fit a reverse diffusion-style Markov chain where each step samples an index
   permutation from GPL and applies it to the current permutation. The strongest
   version uses a separate free GPL score matrix S[step, current_state, D, D].

Permutation convention:
    A permutation "0123" means node_at_position = [0, 1, 2, 3].

GPL probability:
    P(pi | S) = product_t exp(S[t, pi_t]) /
        sum_{j not selected before t} exp(S[t, j]).
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
import torch


Perm = Tuple[int, ...]


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def all_perms(num_nodes: int) -> List[Perm]:
    return list(itertools.permutations(range(num_nodes)))


def perm_to_str(perm: Perm) -> str:
    return "".join(str(x) for x in perm)


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
        if num_nodes != 4:
            raise ValueError("four_modes is currently defined only for num_nodes=4.")
        modes = [
            tuple(range(num_nodes)),
            (1, 0, 2, 3),
            (2, 3, 0, 1),
            tuple(reversed(range(num_nodes))),
        ]
        for mode in modes:
            p[perm_to_idx[mode]] = 0.25
    elif target_name == "topo_0to2_1to2":
        valid = [idx for idx, perm in enumerate(perms) if is_valid_topological_order(perm, [(0, 2), (1, 2)])]
        p[valid] = 1.0 / len(valid)
    elif target_name == "cpdag_chain_uniform_orders":
        if num_nodes != 4:
            raise ValueError("cpdag_chain_uniform_orders is defined for num_nodes=4.")
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
        raise RuntimeError(f"Target {target_name!r} sums to {p.sum()}, not 1.")
    return p


def build_perm_tensor(perms: Sequence[Perm], device: torch.device) -> torch.Tensor:
    return torch.tensor(perms, dtype=torch.long, device=device)


def gpl_log_probs(scores: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
    """
    Exact GPL log probabilities for all permutations.

    Args:
        scores: [..., D, D], where scores[..., t, j] scores choosing node j at
            position t.
        perms: [P, D], all node_at_position permutations.

    Returns:
        log_probs: [..., P]
    """
    num_perms, num_nodes = perms.shape
    batch_shape = scores.shape[:-2]
    flat_scores = scores.reshape(-1, num_nodes, num_nodes)
    out = []
    for score in flat_scores:
        selected = torch.zeros((num_perms, num_nodes), dtype=torch.bool, device=scores.device)
        logp = torch.zeros(num_perms, dtype=scores.dtype, device=scores.device)
        for t in range(num_nodes):
            logits = score[t].unsqueeze(0).expand(num_perms, -1).masked_fill(selected, -torch.inf)
            chosen = perms[:, t]
            logp = logp + torch.log_softmax(logits, dim=-1)[torch.arange(num_perms, device=scores.device), chosen]
            selected_next = selected.clone()
            selected_next[torch.arange(num_perms, device=scores.device), chosen] = True
            selected = selected_next
        out.append(logp)
    return torch.stack(out, dim=0).reshape(*batch_shape, num_perms)


def tv_distance_np(target: np.ndarray, probs: np.ndarray) -> float:
    return float(0.5 * np.abs(target - probs).sum())


def kl_np(target: np.ndarray, probs: np.ndarray, eps: float = 1e-12) -> float:
    mask = target > 0
    return float((target[mask] * (np.log(target[mask] + eps) - np.log(probs[mask] + eps))).sum())


def entropy_np(probs: np.ndarray, eps: float = 1e-12) -> float:
    mask = probs > 0
    return float(-(probs[mask] * np.log(probs[mask] + eps)).sum())


def support_mass_np(target: np.ndarray, probs: np.ndarray) -> float:
    return float(probs[target > 0].sum())


def metric_dict(target: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    return {
        "tv": tv_distance_np(target, probs),
        "kl_target_to_model": kl_np(target, probs),
        "target_support_mass": support_mass_np(target, probs),
        "entropy": entropy_np(probs),
    }


def cross_entropy_loss(target: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
    return -(target * log_probs).sum(dim=-1)


def optimize_one_step(
    *,
    target_np: np.ndarray,
    perms: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed)
    target = torch.tensor(target_np, dtype=torch.float32, device=device)
    best_loss = math.inf
    best_scores = None
    best_probs = None

    for restart in range(args.restarts):
        scores = torch.randn((args.num_nodes, args.num_nodes), dtype=torch.float32, device=device) * args.init_scale
        scores.requires_grad_(True)
        opt = torch.optim.Adam([scores], lr=args.lr)
        for step in range(args.steps):
            opt.zero_grad()
            log_probs = gpl_log_probs(scores, perms)
            loss = cross_entropy_loss(target, log_probs)
            loss.backward()
            opt.step()
            if args.center_scores:
                with torch.no_grad():
                    # Adding a constant to all node scores at the same position
                    # does not change that step's softmax. Column centering would
                    # change the GPL distribution, so only center rows.
                    scores -= scores.mean(dim=1, keepdim=True)

        with torch.no_grad():
            log_probs = gpl_log_probs(scores, perms)
            loss = float(cross_entropy_loss(target, log_probs).item())
            probs = torch.exp(log_probs).detach().cpu().numpy()
            if loss < best_loss:
                best_loss = loss
                best_scores = scores.detach().cpu().numpy()
                best_probs = probs
        print(f"  one_step restart {restart + 1}/{args.restarts}: CE={loss:.6f}, TV={tv_distance_np(target_np, probs):.6f}")

    assert best_scores is not None and best_probs is not None
    return best_scores, best_probs, best_loss


def build_apply_index(perms_list: Sequence[Perm]) -> np.ndarray:
    """apply_index[state_idx, index_perm_idx] = next_state_idx."""
    state_to_idx = {perm: idx for idx, perm in enumerate(perms_list)}
    out = np.empty((len(perms_list), len(perms_list)), dtype=np.int64)
    for s_idx, state in enumerate(perms_list):
        for r_idx, index_perm in enumerate(perms_list):
            next_state = tuple(state[i] for i in index_perm)
            out[s_idx, r_idx] = state_to_idx[next_state]
    return out


def propagate_multistep(
    scores: torch.Tensor,
    perms: torch.Tensor,
    apply_index: torch.Tensor,
    conditioning: str,
) -> torch.Tensor:
    """
    Propagate p_T through reverse GPL transitions.

    Args:
        scores:
            global: [K, D, D]
            state:  [K, P, D, D]
        perms: [P, D], index permutations.
        apply_index: [P, P], next state for current state and sampled index perm.
    """
    num_states = apply_index.size(0)
    p = torch.full((num_states,), 1.0 / num_states, dtype=scores.dtype, device=scores.device)
    num_steps = scores.size(0)
    for k in range(num_steps):
        next_p = torch.zeros_like(p)
        if conditioning == "global":
            trans = torch.exp(gpl_log_probs(scores[k], perms))  # [P_index]
            contrib = p[:, None] * trans[None, :]
        elif conditioning == "state":
            trans = torch.exp(gpl_log_probs(scores[k], perms))  # [P_state, P_index]
            contrib = p[:, None] * trans
        else:
            raise ValueError(f"Unknown conditioning {conditioning!r}.")
        next_p.scatter_add_(0, apply_index.reshape(-1), contrib.reshape(-1))
        p = next_p
    return p


def optimize_multistep(
    *,
    target_np: np.ndarray,
    perms: torch.Tensor,
    apply_index: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed)
    target = torch.tensor(target_np, dtype=torch.float32, device=device)
    num_states = perms.size(0)
    best_loss = math.inf
    best_scores = None
    best_probs = None

    if args.conditioning == "global":
        score_shape = (args.num_steps, args.num_nodes, args.num_nodes)
    elif args.conditioning == "state":
        score_shape = (args.num_steps, num_states, args.num_nodes, args.num_nodes)
    else:
        raise ValueError(f"Unknown conditioning {args.conditioning!r}.")

    for restart in range(args.restarts):
        scores = torch.randn(score_shape, dtype=torch.float32, device=device) * args.init_scale
        scores.requires_grad_(True)
        opt = torch.optim.Adam([scores], lr=args.lr)
        for step in range(args.steps):
            opt.zero_grad()
            probs = propagate_multistep(scores, perms, apply_index, conditioning=args.conditioning)
            loss = -(target * torch.log(probs.clamp_min(1e-12))).sum()
            loss.backward()
            opt.step()
            if args.center_scores:
                with torch.no_grad():
                    # Last two dims are [position, node]. Only row shifts over
                    # nodes at a fixed position are GPL-invariant.
                    scores -= scores.mean(dim=-1, keepdim=True)

        with torch.no_grad():
            probs_t = propagate_multistep(scores, perms, apply_index, conditioning=args.conditioning)
            loss = float((-(target * torch.log(probs_t.clamp_min(1e-12))).sum()).item())
            probs = probs_t.detach().cpu().numpy()
            if loss < best_loss:
                best_loss = loss
                best_scores = scores.detach().cpu().numpy()
                best_probs = probs
        print(
            f"  multistep restart {restart + 1}/{args.restarts}: "
            f"CE={loss:.6f}, TV={tv_distance_np(target_np, probs):.6f}"
        )

    assert best_scores is not None and best_probs is not None
    return best_scores, best_probs, best_loss


def distribution_rows(target_name: str, seed: int, perms_list: Sequence[Perm], target: np.ndarray, probs: np.ndarray):
    rows = []
    for idx, perm in enumerate(perms_list):
        rows.append(
            {
                "target": target_name,
                "seed": seed,
                "perm": perm_to_str(perm),
                "p_target": float(target[idx]),
                "p_model": float(probs[idx]),
            }
        )
    return rows


def score_rows(target_name: str, seed: int, scores: np.ndarray, mode: str, conditioning: str):
    rows = []
    if mode == "one_step":
        for pos in range(scores.shape[0]):
            for node in range(scores.shape[1]):
                rows.append(
                    {
                        "target": target_name,
                        "seed": seed,
                        "mode": mode,
                        "conditioning": "",
                        "step": 0,
                        "state_idx": -1,
                        "position": pos,
                        "node": node,
                        "score": float(scores[pos, node]),
                    }
                )
    elif conditioning == "global":
        for step in range(scores.shape[0]):
            for pos in range(scores.shape[1]):
                for node in range(scores.shape[2]):
                    rows.append(
                        {
                            "target": target_name,
                            "seed": seed,
                            "mode": mode,
                            "conditioning": conditioning,
                            "step": step,
                            "state_idx": -1,
                            "position": pos,
                            "node": node,
                            "score": float(scores[step, pos, node]),
                        }
                    )
    else:
        for step in range(scores.shape[0]):
            for state_idx in range(scores.shape[1]):
                for pos in range(scores.shape[2]):
                    for node in range(scores.shape[3]):
                        rows.append(
                            {
                                "target": target_name,
                                "seed": seed,
                                "mode": mode,
                                "conditioning": conditioning,
                                "step": step,
                                "state_idx": state_idx,
                                "position": pos,
                                "node": node,
                                "score": float(scores[step, state_idx, pos, node]),
                            }
                        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPL permutation expressivity experiments.")
    parser.add_argument("--mode", type=str, default="one_step", choices=["one_step", "multistep"])
    parser.add_argument("--num_nodes", type=int, default=4)
    parser.add_argument(
        "--targets",
        type=str,
        default="single,close_two,opposite_two,cpdag_chain_uniform_orders",
        help=(
            "Comma-separated targets: single, close_two, opposite_two, four_modes, "
            "topo_0to2_1to2, cpdag_chain_uniform_orders, cpdag_chain_uniform_dags."
        ),
    )
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--num_steps", type=int, default=3)
    parser.add_argument("--conditioning", type=str, default="state", choices=["global", "state"])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--init_scale", type=float, default=0.01)
    parser.add_argument("--center_scores", action="store_true", default=True)
    parser.add_argument("--no_center_scores", action="store_false", dest="center_scores")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--results_dir", type=str, default="all_experiments/gpl_expressivity_results")
    parser.add_argument("--top_k_print", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_nodes > 5:
        raise ValueError("This script enumerates D! permutations; use num_nodes <= 5.")
    device = torch.device(args.device)
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    perms_list = all_perms(args.num_nodes)
    perms = build_perm_tensor(perms_list, device=device)
    apply_index_np = build_apply_index(perms_list)
    apply_index = torch.tensor(apply_index_np, dtype=torch.long, device=device)
    targets = parse_csv_list(args.targets)
    seeds = [int(x) for x in parse_csv_list(args.seeds)]

    print("=" * 100)
    print("GPL expressivity experiment")
    print(f"mode:          {args.mode}")
    print(f"num_nodes:     {args.num_nodes}")
    print(f"num_perms:     {len(perms_list)}")
    print(f"targets:       {targets}")
    print(f"seeds:         {seeds}")
    if args.mode == "multistep":
        print(f"num_steps:     {args.num_steps}")
        print(f"conditioning:  {args.conditioning}")
    print(f"steps/restarts:{args.steps}/{args.restarts}")
    print(f"results_dir:   {results_dir}")
    print("=" * 100)

    summary_rows: List[Dict[str, object]] = []
    dist_rows: List[Dict[str, object]] = []
    score_out_rows: List[Dict[str, object]] = []
    start = time.time()

    for target_name in targets:
        target = make_target_distribution(target_name, perms_list, args.num_nodes)
        for seed in seeds:
            print("-" * 100)
            print(f"target={target_name}, seed={seed}")
            if args.mode == "one_step":
                scores, probs, ce = optimize_one_step(
                    target_np=target,
                    perms=perms,
                    args=args,
                    seed=seed,
                    device=device,
                )
            else:
                scores, probs, ce = optimize_multistep(
                    target_np=target,
                    perms=perms,
                    apply_index=apply_index,
                    args=args,
                    seed=seed,
                    device=device,
                )

            m = metric_dict(target, probs)
            summary = {
                "target": target_name,
                "seed": seed,
                "mode": args.mode,
                "conditioning": args.conditioning if args.mode == "multistep" else "",
                "num_nodes": args.num_nodes,
                "num_perms": len(perms_list),
                "num_steps": args.num_steps if args.mode == "multistep" else 1,
                "optimization_steps": args.steps,
                "restarts": args.restarts,
                "lr": args.lr,
                "cross_entropy": ce,
                "tv": m["tv"],
                "kl_target_to_model": m["kl_target_to_model"],
                "target_support_mass": m["target_support_mass"],
                "entropy": m["entropy"],
                "scores_json": json.dumps(scores.tolist()),
            }
            summary_rows.append(summary)
            dist_rows.extend(distribution_rows(target_name, seed, perms_list, target, probs))
            score_out_rows.extend(score_rows(target_name, seed, scores, args.mode, args.conditioning))

            print(
                f"RESULT target={target_name} seed={seed}: "
                f"TV={m['tv']:.6f}, support={m['target_support_mass']:.6f}, "
                f"KL={m['kl_target_to_model']:.6f}"
            )
            top_idx = np.argsort(-np.maximum(target, probs))[: args.top_k_print]
            for idx in top_idx:
                print(f"  {perm_to_str(perms_list[idx])}: target={target[idx]:.4f}, model={probs[idx]:.4f}")

    summary_df = pd.DataFrame(summary_rows)
    dist_df = pd.DataFrame(dist_rows)
    scores_df = pd.DataFrame(score_out_rows)

    suffix = args.mode if args.mode == "one_step" else f"multistep_{args.conditioning}_K{args.num_steps}"
    summary_path = results_dir / f"summary_gpl_{suffix}.csv"
    dist_path = results_dir / f"permutation_distributions_gpl_{suffix}.csv"
    scores_path = results_dir / f"best_scores_gpl_{suffix}.csv"
    summary_df.to_csv(summary_path, index=False)
    dist_df.to_csv(dist_path, index=False)
    scores_df.to_csv(scores_path, index=False)

    print("=" * 100)
    print(f"Finished in {time.time() - start:.1f}s")
    print(f"Wrote summary:       {summary_path}")
    print(f"Wrote distributions: {dist_path}")
    print(f"Wrote scores:        {scores_path}")


if __name__ == "__main__":
    main()
