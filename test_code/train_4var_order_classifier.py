#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a 4-node topological-order upper-bound classifier.

The model enumerates all 4! = 24 node orders and learns to put probability mass
on the orders that are valid topological orders of the label DAG.  It does not
use priority u, diffusion, Gumbel-Sinkhorn, or an edge decoder.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    edge_batch, edge_parent, edge_child = torch.nonzero(graph, as_tuple=True)
    if edge_batch.numel() == 0:
        return valid

    parent_pos = positions[:, edge_parent].transpose(0, 1)
    child_pos = positions[:, edge_child].transpose(0, 1)
    edge_valid = parent_pos < child_pos
    valid[edge_batch] &= edge_valid
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


class FourNodeOrderClassifier(CausalTNPEncoder):
    """Causal encoder plus direct scorer over all D! node orders."""

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
        hidden = scorer_hidden or dim_feedforward
        self.order_scorer = nn.Sequential(
            nn.Linear(num_nodes * d_model, hidden, bias=True, device=device, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden, bias=True, device=device, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1, bias=True, device=device, dtype=dtype),
        )

    def forward(self, target_data: Tensor, orders: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if target_data.dim() == 3:
            target_data = target_data.unsqueeze(-1)
        node_repr = self.encode(target_data=target_data, mask=mask).squeeze(2)
        batch_size, num_nodes, d_model = node_repr.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {num_nodes}.")

        num_orders = orders.size(0)
        gather_idx = orders.view(1, num_orders, num_nodes, 1).expand(batch_size, -1, -1, d_model)
        node_repr_rep = node_repr.unsqueeze(1).expand(-1, num_orders, -1, -1)
        ordered_repr = torch.gather(node_repr_rep, 2, gather_idx)
        score_input = ordered_repr.reshape(batch_size, num_orders, num_nodes * d_model)
        return self.order_scorer(score_input).squeeze(-1)


def order_mass_loss(scores: Tensor, valid_mask: Tensor) -> Tensor:
    """-log probability assigned to any valid topological order."""
    neg_inf = torch.finfo(scores.dtype).min
    valid_scores = scores.masked_fill(~valid_mask, neg_inf)
    return -(torch.logsumexp(valid_scores.float(), dim=-1) - torch.logsumexp(scores.float(), dim=-1))


def evaluate(
    model: FourNodeOrderClassifier,
    loader: torch.utils.data.DataLoader,
    *,
    orders: Tensor,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    total = 0
    sums: Dict[str, float] = {
        "loss": 0.0,
        "top1_valid_rate": 0.0,
        "valid_mass": 0.0,
        "edge_precedence_accuracy": 0.0,
        "num_valid_orders": 0.0,
    }

    with torch.no_grad():
        for batch_idx, (inputs, targets, mask) in enumerate(tqdm(loader, desc="Eval", leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.float32) if mask is not None else None

            scores = model(inputs, orders=orders, mask=mask)
            valid = valid_order_mask(targets, orders)
            loss = order_mass_loss(scores, valid)
            probs = torch.softmax(scores.float(), dim=-1)
            top_idx = scores.argmax(dim=-1)
            top_orders = orders[top_idx]

            batch_size = inputs.size(0)
            total += batch_size
            sums["loss"] += float(loss.sum().item())
            sums["top1_valid_rate"] += float(valid[torch.arange(batch_size, device=device), top_idx].float().sum().item())
            sums["valid_mass"] += float((probs * valid.float()).sum(dim=-1).sum().item())
            sums["edge_precedence_accuracy"] += float(edge_precedence_accuracy(targets, top_orders).sum().item())
            sums["num_valid_orders"] += float(valid.float().sum(dim=-1).sum().item())

    return {key: value / max(total, 1) for key, value in sums.items()}


def train_epoch(
    model: FourNodeOrderClassifier,
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
    sums: Dict[str, float] = {
        "loss": 0.0,
        "top1_valid_rate": 0.0,
        "valid_mass": 0.0,
        "edge_precedence_accuracy": 0.0,
        "num_valid_orders": 0.0,
    }

    for batch_idx, (inputs, targets, mask) in enumerate(tqdm(loader, desc="Train", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs = inputs.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device, dtype=torch.float32) if mask is not None else None

        scores = model(inputs, orders=orders, mask=mask)
        valid = valid_order_mask(targets, orders)
        loss_per_graph = order_mass_loss(scores, valid)
        loss = loss_per_graph.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            probs = torch.softmax(scores.float(), dim=-1)
            top_idx = scores.argmax(dim=-1)
            top_orders = orders[top_idx]
            batch_size = inputs.size(0)
            total += batch_size
            sums["loss"] += float(loss_per_graph.sum().item())
            sums["top1_valid_rate"] += float(valid[torch.arange(batch_size, device=device), top_idx].float().sum().item())
            sums["valid_mass"] += float((probs * valid.float()).sum(dim=-1).sum().item())
            sums["edge_precedence_accuracy"] += float(edge_precedence_accuracy(targets, top_orders).sum().item())
            sums["num_valid_orders"] += float(valid.float().sum(dim=-1).sum().item())

    return {key: value / max(total, 1) for key, value in sums.items()}


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 4-node direct topological-order classifier.")
    parser.add_argument("--work_dir", type=str, default="/home/rtyen/projects/CausalStructureNeuralProcess-main/ml2_meta_causal_discovery")
    parser.add_argument("--synth_data_root", type=str, default=None, help="Optional root containing synthetic training datasets.")
    parser.add_argument("--data_file", type=str, default="gp_4var_ERL0U1")
    parser.add_argument("--run_name", type=str, default="order_classifier_4var_d128")
    parser.add_argument("--results_dir", type=str, default="result/order_classifier_4var")

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
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    if args.num_nodes != 4:
        raise ValueError("This upper-bound script is intended for exactly 4 nodes.")

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

    model = FourNodeOrderClassifier(
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
    orders = all_permutation_orders(args.num_nodes, device=device)
    print(f"Enumerating {orders.size(0)} orders:")
    print(orders.detach().cpu().tolist())

    run_dir = Path(args.results_dir).expanduser().resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    rows: List[Dict[str, float]] = []
    best_val = math.inf
    for epoch in range(args.max_epochs):
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
        pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)

        torch.save(model.state_dict(), run_dir / f"model_{epoch}.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_top1_valid={train_metrics['top1_valid_rate']:.4f} "
            f"train_valid_mass={train_metrics['valid_mass']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_top1_valid={val_metrics['top1_valid_rate']:.4f} "
            f"val_valid_mass={val_metrics['valid_mass']:.4f}"
        )

    print(f"Wrote run outputs to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
