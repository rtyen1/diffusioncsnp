#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a one-step Plackett-Luce model for 4-node topological orders.

This is an order-only baseline.  The model predicts one scalar score per node,
which defines a standard PL distribution over all 4! orders.  The target
distribution is uniform over all valid root-to-leaf topological orders of the
label DAG.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from ml2_meta_causal_discovery.models.causaltransformercomponents import CausalTNPEncoder
from ml2_meta_causal_discovery.utils.datautils import (
    MultipleFileDatasetWithPadding,
    transformer_classifier_split_withpadding,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def all_permutation_orders(num_nodes: int, device: torch.device) -> Tensor:
    orders = list(itertools.permutations(range(num_nodes)))
    return torch.tensor(orders, dtype=torch.long, device=device)


def valid_order_mask(graph: Tensor, orders: Tensor) -> Tensor:
    """Return [batch, num_orders] mask for root-to-leaf valid orders."""
    graph = (graph > 0.5).to(torch.bool)
    batch_size, num_nodes, _ = graph.shape
    num_orders = orders.size(0)

    positions = torch.empty((num_orders, num_nodes), dtype=torch.long, device=orders.device)
    order_ids = torch.arange(num_orders, device=orders.device).unsqueeze(1).expand(-1, num_nodes)
    pos_ids = torch.arange(num_nodes, device=orders.device).unsqueeze(0).expand(num_orders, -1)
    positions[order_ids, orders] = pos_ids

    valid = torch.ones((batch_size, num_orders), dtype=torch.bool, device=graph.device)
    for b in range(batch_size):
        edges = torch.nonzero(graph[b], as_tuple=False)
        if edges.numel() == 0:
            continue
        parent_pos = positions[:, edges[:, 0]]
        child_pos = positions[:, edges[:, 1]]
        valid[b] = (parent_pos < child_pos).all(dim=1)
    return valid


def edge_precedence_accuracy(graph: Tensor, orders: Tensor) -> Tensor:
    """Top-1 edge precedence accuracy for each graph in a batch."""
    graph = (graph > 0.5).to(torch.bool)
    batch_size, num_nodes, _ = graph.shape
    positions = torch.empty((batch_size, num_nodes), dtype=torch.long, device=orders.device)
    batch_idx = torch.arange(batch_size, device=orders.device).unsqueeze(1).expand(-1, num_nodes)
    pos_idx = torch.arange(num_nodes, device=orders.device).unsqueeze(0).expand(batch_size, -1)
    positions[batch_idx, orders] = pos_idx

    out = torch.ones(batch_size, dtype=torch.float32, device=orders.device)
    for b in range(batch_size):
        edges = torch.nonzero(graph[b], as_tuple=False)
        if edges.numel() == 0:
            out[b] = 1.0
            continue
        ok = positions[b, edges[:, 0]] < positions[b, edges[:, 1]]
        out[b] = ok.float().mean()
    return out


def pl_log_prob(scores: Tensor, orders: Tensor) -> Tensor:
    """Log P(order | scores) under a standard Plackett-Luce distribution.

    Args:
        scores: [batch, num_nodes]
        orders: [num_orders, num_nodes]

    Returns:
        [batch, num_orders]
    """
    batch_size = scores.size(0)
    num_orders, num_nodes = orders.shape
    ordered_scores = scores.index_select(1, orders.reshape(-1)).view(
        batch_size,
        num_orders,
        num_nodes,
    )

    log_terms = []
    for pos in range(num_nodes):
        chosen = ordered_scores[:, :, pos]
        remaining_logsumexp = torch.logsumexp(ordered_scores[:, :, pos:], dim=-1)
        log_terms.append(chosen - remaining_logsumexp)
    return torch.stack(log_terms, dim=-1).sum(dim=-1)


def uniform_valid_distribution(valid: Tensor) -> Tensor:
    valid_float = valid.float()
    num_valid = valid_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return valid_float / num_valid


def pl_uniform_loss(scores: Tensor, graph: Tensor, orders: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    log_probs = pl_log_prob(scores.float(), orders)
    log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
    valid = valid_order_mask(graph, orders)
    target = uniform_valid_distribution(valid)
    loss = -(target * log_probs).sum(dim=-1)
    return loss, log_probs, valid, target


def summarize_batch(
    loss_per_graph: Tensor,
    log_probs: Tensor,
    valid: Tensor,
    target: Tensor,
    graph: Tensor,
    orders: Tensor,
) -> Dict[str, Tensor]:
    probs = log_probs.exp()
    top_idx = log_probs.argmax(dim=-1)
    top_orders = orders[top_idx]
    batch_idx = torch.arange(log_probs.size(0), device=log_probs.device)

    num_valid = valid.float().sum(dim=-1)
    entropy_uniform = num_valid.clamp_min(1.0).log()
    kl_uniform = loss_per_graph - entropy_uniform
    tv_uniform = 0.5 * (probs - target).abs().sum(dim=-1)

    return {
        "loss": loss_per_graph,
        "kl_uniform": kl_uniform,
        "tv_uniform": tv_uniform,
        "valid_mass": (probs * valid.float()).sum(dim=-1),
        "top1_valid_rate": valid[batch_idx, top_idx].float(),
        "edge_precedence_accuracy": edge_precedence_accuracy(graph, top_orders),
        "num_valid_orders": num_valid,
        "entropy_uniform": entropy_uniform,
    }


class FourNodePLOneStepModel(CausalTNPEncoder):
    """Causal encoder plus one scalar PL score per node."""

    def __init__(
        self,
        *,
        d_model: int,
        dim_feedforward: int,
        nhead: int,
        num_layers_encoder: int,
        num_nodes: int,
        emb_depth: int = 1,
        dropout: float = 0.0,
        scorer_hidden: Optional[int] = None,
        use_positional_encoding: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            num_layers=num_layers_encoder,
            emb_depth=emb_depth,
            num_nodes=num_nodes,
            use_positional_encoding=use_positional_encoding,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        self.num_nodes = num_nodes
        hidden = scorer_hidden or d_model
        self.node_scorer = nn.Sequential(
            nn.Linear(d_model, hidden, bias=True, device=device, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1, bias=True, device=device, dtype=dtype),
        )

    def forward(self, target_data: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if target_data.dim() == 3:
            target_data = target_data.unsqueeze(-1)
        node_repr = self.encode(target_data=target_data, mask=mask).squeeze(2)
        if node_repr.size(1) != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {node_repr.size(1)}.")
        return self.node_scorer(node_repr).squeeze(-1)


def update_sums(sums: Dict[str, float], batch_metrics: Dict[str, Tensor]) -> None:
    for key, value in batch_metrics.items():
        sums[key] += float(value.sum().detach().cpu().item())


def average_sums(sums: Dict[str, float], total: int) -> Dict[str, float]:
    return {key: value / max(total, 1) for key, value in sums.items()}


def empty_metric_sums() -> Dict[str, float]:
    return {
        "loss": 0.0,
        "kl_uniform": 0.0,
        "tv_uniform": 0.0,
        "valid_mass": 0.0,
        "top1_valid_rate": 0.0,
        "edge_precedence_accuracy": 0.0,
        "num_valid_orders": 0.0,
        "entropy_uniform": 0.0,
    }


def evaluate(
    model: FourNodePLOneStepModel,
    loader: torch.utils.data.DataLoader,
    *,
    orders: Tensor,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    total = 0
    sums = empty_metric_sums()

    with torch.no_grad():
        for batch_idx, (inputs, targets, mask) in enumerate(tqdm(loader, desc="Eval", leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.float32) if mask is not None else None

            scores = model(inputs, mask=mask)
            loss_per_graph, log_probs, valid, target = pl_uniform_loss(scores, targets, orders)
            metrics = summarize_batch(loss_per_graph, log_probs, valid, target, targets, orders)

            total += inputs.size(0)
            update_sums(sums, metrics)

    return average_sums(sums, total)


def train_epoch(
    model: FourNodePLOneStepModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    orders: Tensor,
    device: torch.device,
    grad_clip: Optional[float],
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    total = 0
    sums = empty_metric_sums()

    for batch_idx, (inputs, targets, mask) in enumerate(tqdm(loader, desc="Train", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs = inputs.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device, dtype=torch.float32) if mask is not None else None

        scores = model(inputs, mask=mask)
        loss_per_graph, log_probs, valid, target = pl_uniform_loss(scores, targets, orders)
        loss = loss_per_graph.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            metrics = summarize_batch(loss_per_graph, log_probs, valid, target, targets, orders)
            total += inputs.size(0)
            update_sums(sums, metrics)

    return average_sums(sums, total)


def make_loader(
    split_dir: Path,
    *,
    batch_size: int,
    num_workers: int,
    num_nodes: int,
    sample_size_min: int,
    sample_size_max: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    files = sorted(split_dir.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No hdf5 files found in {split_dir}")
    dataset = MultipleFileDatasetWithPadding(files, max_node_num=num_nodes)
    collator = transformer_classifier_split_withpadding(
        sample_size_min=sample_size_min,
        sample_size_max=sample_size_max,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False if num_workers == 0 else True,
        collate_fn=collator,
    )


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def maybe_resume(
    checkpoint_path: Optional[str],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    if not checkpoint_path:
        return 0
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Resume checkpoint must contain model_state_dict.")
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device=device)
    start_epoch = int(checkpoint.get("epoch", 0))
    print(f"Resumed from {checkpoint_path} at epoch {start_epoch}.")
    return start_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one-step PL order model with uniform-valid targets.")
    parser.add_argument("--work_dir", type=str, default="/home/rtyen/projects/CausalStructureNeuralProcess-main/ml2_meta_causal_discovery")
    parser.add_argument("--synth_data_root", type=str, default=None, help="Optional root containing synthetic training datasets.")
    parser.add_argument("--data_file", type=str, default="gp_4var_ERL0U1")
    parser.add_argument("--run_name", type=str, default="pl_order_uniform_4var_d128")
    parser.add_argument("--results_dir", type=str, default="result/pl_order_4var")

    parser.add_argument("--num_nodes", type=int, default=4)
    parser.add_argument("--dim_model", type=int, default=128)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers_encoder", type=int, default=4)
    parser.add_argument("--scorer_hidden", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_positional_encoding", action="store_true")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--sample_size_min", type=int, default=100)
    parser.add_argument("--sample_size_max", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--eval_max_batches", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    if args.num_nodes != 4:
        raise ValueError("This script is intended for exactly 4 nodes.")

    work_dir = Path(args.work_dir).expanduser().resolve()
    synth_data_root = (
        Path(args.synth_data_root).expanduser().resolve()
        if args.synth_data_root
        else work_dir / "datasets" / "data" / "synth_training_data"
    )
    data_root = synth_data_root / args.data_file
    train_loader = make_loader(
        data_root / "train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_nodes=args.num_nodes,
        sample_size_min=args.sample_size_min,
        sample_size_max=args.sample_size_max,
        shuffle=True,
    )
    val_loader = make_loader(
        data_root / "val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_nodes=args.num_nodes,
        sample_size_min=args.sample_size_min,
        sample_size_max=args.sample_size_max,
        shuffle=False,
    )

    model = FourNodePLOneStepModel(
        d_model=args.dim_model,
        dim_feedforward=args.dim_feedforward,
        nhead=args.nhead,
        num_layers_encoder=args.num_layers_encoder,
        num_nodes=args.num_nodes,
        dropout=args.dropout,
        scorer_hidden=args.scorer_hidden,
        use_positional_encoding=args.use_positional_encoding,
        device=device,
        dtype=torch.float32,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch = maybe_resume(
        args.resume_from_checkpoint,
        model=model,
        optimizer=optimizer,
        device=device,
    )

    orders = all_permutation_orders(args.num_nodes, device=device)
    print(f"Enumerating {orders.size(0)} orders:")
    print(orders.detach().cpu().tolist())

    run_dir = Path(args.results_dir).expanduser().resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    with open(run_dir / "orders.json", "w", encoding="utf-8") as f:
        json.dump(orders.detach().cpu().tolist(), f, indent=2)

    metrics_path = run_dir / "metrics.csv"
    rows: List[Dict[str, float]]
    if start_epoch > 0 and metrics_path.exists():
        rows = pd.read_csv(metrics_path).query("epoch < @start_epoch").to_dict("records")
        best_val = min((float(row["val_kl_uniform"]) for row in rows), default=math.inf)
    else:
        rows = []
        best_val = math.inf

    for epoch in range(start_epoch, args.max_epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            orders=orders,
            device=device,
            grad_clip=args.grad_clip,
            max_batches=args.max_train_batches,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            orders=orders,
            device=device,
            max_batches=args.eval_max_batches,
        )

        row: Dict[str, float] = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        rows.append(row)
        pd.DataFrame(rows).to_csv(metrics_path, index=False)

        torch.save(model.state_dict(), run_dir / f"model_{epoch}.pt")
        save_checkpoint(
            run_dir / f"checkpoint_{epoch}.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            args=args,
        )
        if val_metrics["kl_uniform"] < best_val:
            best_val = val_metrics["kl_uniform"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")

        print(
            f"epoch={epoch:03d} "
            f"train_ce={train_metrics['loss']:.4f} "
            f"train_kl={train_metrics['kl_uniform']:.4f} "
            f"train_tv={train_metrics['tv_uniform']:.4f} "
            f"train_valid_mass={train_metrics['valid_mass']:.4f} "
            f"val_ce={val_metrics['loss']:.4f} "
            f"val_kl={val_metrics['kl_uniform']:.4f} "
            f"val_tv={val_metrics['tv_uniform']:.4f} "
            f"val_valid_mass={val_metrics['valid_mass']:.4f}"
        )

    print(f"Wrote run outputs to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
