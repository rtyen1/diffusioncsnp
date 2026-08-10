"""
Order-only topological sorting model using the SymmetricDiffusers diffusion core.

This module keeps the permutation diffusion objective from SymmetricDiffusers and
only adapts the object embedding stage: causal variables are embedded with the
existing CausalTNPEncoder instead of an image/TSP encoder.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ml2_meta_causal_discovery.models.causaltransformercomponents import (
    CausalAdjacencyMatrix,
    CausalTNPEncoder,
    CausalTransformerDecoderLayer,
)
from ml2_meta_causal_discovery.utils.permutations import sample_permutation
from ml2_meta_causal_discovery.utils.topological_orders import (
    priority_kahn_topological_sort,
    random_kahn_topological_sort,
)


_SYMMETRIC_DIFFUSERS_DIR = Path(__file__).resolve().parents[2] / "SymmetricDiffusers"
if str(_SYMMETRIC_DIFFUSERS_DIR) not in sys.path:
    sys.path.insert(0, str(_SYMMETRIC_DIFFUSERS_DIR))

import utils as _sd_utils  # noqa: E402,F401
import PL_distribution as PL  # noqa: E402
from diffusion import DiffusionUtils  # noqa: E402
from models import EncoderLayers  # noqa: E402
from models import TimestepEmbedder  # noqa: E402


class BakStyleSkeletonHead(nn.Module):
    """Bak-style symmetric skeleton head.

    Input node representations keep the original node id order. The output
    logits are symmetric and are intended for unordered skeleton prediction.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.decoder = nn.TransformerDecoder(
            decoder_layer=CausalTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,
                batch_first=True,
                device=device,
                dtype=dtype,
                bias=False,
            ),
            num_layers=num_layers,
        )
        self.param = CausalAdjacencyMatrix(
            nhead=nhead,
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

    def forward(self, node_repr: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        skel_repr = self.decoder(
            node_repr,
            memory=None,
            tgt_key_padding_mask=padding_mask,
        )
        logits = self.param(skel_repr, padding_mask=None)
        return (logits + logits.transpose(1, 2)) / 2


class BakStyleSkeletonMixin:
    def _init_skeleton_head(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers_skeleton: int,
        dropout: float,
        skeleton_loss_weight: float,
        order_loss_weight: float,
        device=None,
        dtype=None,
    ) -> None:
        self.skeleton_loss_weight = skeleton_loss_weight
        self.order_loss_weight = order_loss_weight
        self.skeleton_head = BakStyleSkeletonHead(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers_skeleton,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _decoder_mask(mask: Optional[Tensor]) -> Optional[Tensor]:
        return mask[:, 0, :] if mask is not None else None

    def _skeleton_logits_from_data(self, target_data: Tensor, mask: Optional[Tensor]) -> Tensor:
        raw_node_repr = self._encode_raw_data(target_data, mask=mask)
        return self._skeleton_logits_from_node_repr(raw_node_repr, mask=mask)

    def _skeleton_logits_from_node_repr(
        self,
        node_repr: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        return self.skeleton_head(
            node_repr,
            padding_mask=self._decoder_mask(mask),
        )

    def _skeleton_loss_per_batch(
        self,
        logits: Tensor,
        graph: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        target = ((graph > 0.5) | (graph.transpose(1, 2) > 0.5)).to(dtype=logits.dtype)
        num_nodes = logits.size(-1)
        pair_mask = torch.triu(
            torch.ones((num_nodes, num_nodes), dtype=torch.bool, device=logits.device),
            diagonal=1,
        ).unsqueeze(0)

        decoder_mask = self._decoder_mask(mask)
        if decoder_mask is not None:
            valid_nodes = decoder_mask > -1e20
            pair_mask = pair_mask & valid_nodes.unsqueeze(1) & valid_nodes.unsqueeze(2)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.float(),
            target.float(),
            reduction="none",
        )
        pair_weight = pair_mask.to(dtype=loss.dtype)
        denom = pair_weight.flatten(1).sum(dim=1).clamp_min(1)
        return (loss * pair_weight).flatten(1).sum(dim=1) / denom


class PrecedenceRelationHead(nn.Module):
    """Swap-consistent three-way relation head for unordered node pairs.

    For each ordered pair (i, j), class 0 means i is an ancestor of j,
    class 1 means j is an ancestor of i, and class 2 means incomparable.
    Swapping i and j swaps the first two logits and leaves the third unchanged.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        self.direction = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim, **factory_kwargs),
            nn.GELU(),
            nn.Linear(hidden_dim, 1, **factory_kwargs),
        )
        self.incomparable = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim, **factory_kwargs),
            nn.GELU(),
            nn.Linear(hidden_dim, 1, **factory_kwargs),
        )

    def forward(self, node_repr: Tensor) -> Tensor:
        node_i = node_repr.unsqueeze(2).expand(-1, -1, node_repr.size(1), -1)
        node_j = node_repr.unsqueeze(1).expand(-1, node_repr.size(1), -1, -1)
        forward_logit = self.direction(torch.cat([node_i, node_j], dim=-1)).squeeze(-1)
        backward_logit = forward_logit.transpose(1, 2)
        symmetric_features = torch.cat(
            [(node_i - node_j).abs(), node_i * node_j],
            dim=-1,
        )
        incomparable_logit = self.incomparable(symmetric_features).squeeze(-1)
        return torch.stack(
            [forward_logit, backward_logit, incomparable_logit],
            dim=-1,
        )


class PrecedenceRelationMixin:
    """DAG partial-order supervision and optional final-beam reranking."""

    def _init_precedence_relation(
        self,
        d_model: int,
        hidden_dim: int,
        loss_weight: float,
        rerank_beta: float,
        device=None,
        dtype=None,
    ) -> None:
        if loss_weight < 0:
            raise ValueError("topo_precedence_loss_weight must be non-negative.")
        if rerank_beta < 0:
            raise ValueError("topo_precedence_rerank_beta must be non-negative.")
        if hidden_dim <= 0:
            raise ValueError("topo_precedence_hidden_dim must be positive.")

        self.topo_precedence_loss_weight = float(loss_weight)
        self.topo_precedence_rerank_beta = float(rerank_beta)
        self.topo_precedence_hidden_dim = int(hidden_dim)
        self.precedence_head = None
        if loss_weight > 0 or rerank_beta > 0:
            self.precedence_head = PrecedenceRelationHead(
                d_model=d_model,
                hidden_dim=hidden_dim,
                device=device,
                dtype=dtype,
            )

    @staticmethod
    def _valid_node_mask(mask: Optional[Tensor], graph: Tensor) -> Tensor:
        if mask is None:
            return torch.ones(
                graph.shape[:2],
                dtype=torch.bool,
                device=graph.device,
            )
        return mask[:, -1, :] > -1e20

    @staticmethod
    def _transitive_closure(graph: Tensor, valid_nodes: Tensor) -> Tensor:
        """Boolean reachability for graph[parent, child], excluding padding."""
        reach = graph > 0.5
        valid_pairs = valid_nodes.unsqueeze(2) & valid_nodes.unsqueeze(1)
        reach = reach & valid_pairs
        for k in range(graph.size(-1)):
            reach = reach | (
                reach[:, :, k].unsqueeze(2) & reach[:, k, :].unsqueeze(1)
            )
        diagonal = torch.eye(
            graph.size(-1),
            dtype=torch.bool,
            device=graph.device,
        ).unsqueeze(0)
        return reach & ~diagonal & valid_pairs

    def _precedence_loss_per_batch(
        self,
        logits: Tensor,
        graph: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        valid_nodes = self._valid_node_mask(mask, graph)
        reach = self._transitive_closure(graph, valid_nodes)
        num_nodes = graph.size(-1)
        pair_mask = torch.triu(
            torch.ones(
                (num_nodes, num_nodes),
                dtype=torch.bool,
                device=graph.device,
            ),
            diagonal=1,
        ).unsqueeze(0)
        pair_mask = pair_mask & valid_nodes.unsqueeze(2) & valid_nodes.unsqueeze(1)

        labels = torch.full(
            graph.shape,
            2,
            dtype=torch.long,
            device=graph.device,
        )
        labels = torch.where(reach, torch.zeros_like(labels), labels)
        labels = torch.where(reach.transpose(1, 2), torch.ones_like(labels), labels)
        losses = F.cross_entropy(
            logits.float().permute(0, 3, 1, 2),
            labels,
            reduction="none",
        )
        pair_weight = pair_mask.to(dtype=losses.dtype)
        denom = pair_weight.flatten(1).sum(dim=1).clamp_min(1)
        return (losses * pair_weight).flatten(1).sum(dim=1) / denom

    def _precedence_outputs(
        self,
        raw_node_repr: Tensor,
        graph: Tensor,
        mask: Optional[Tensor],
    ) -> Tuple[Optional[Tensor], Tensor]:
        if self.precedence_head is None or self.topo_precedence_loss_weight == 0:
            zero = torch.zeros(
                raw_node_repr.size(0),
                dtype=raw_node_repr.dtype,
                device=raw_node_repr.device,
            )
            return None, zero
        logits = self.precedence_head(raw_node_repr)
        loss = self._precedence_loss_per_batch(logits, graph=graph, mask=mask)
        return logits, loss

    def _precedence_candidate_scores(
        self,
        relation_logits: Tensor,
        candidates: Tensor,
        valid_nodes: Optional[Tensor] = None,
    ) -> Tensor:
        """Average log compatibility of complete candidate permutations."""
        batch_size, _, num_nodes = candidates.shape
        pair_indices = torch.triu_indices(
            num_nodes,
            num_nodes,
            offset=1,
            device=candidates.device,
        )
        left_nodes = candidates[:, :, pair_indices[0]]
        right_nodes = candidates[:, :, pair_indices[1]]
        batch_idx = torch.arange(batch_size, device=candidates.device)[:, None, None]
        pair_logits = relation_logits[batch_idx, left_nodes, right_nodes]
        pair_probs = F.softmax(pair_logits.float(), dim=-1)
        compatibility = (pair_probs[..., 0] + pair_probs[..., 2]).clamp_min(1e-8)

        if valid_nodes is None:
            pair_weight = torch.ones_like(compatibility)
        else:
            pair_weight = (
                valid_nodes[batch_idx, left_nodes]
                & valid_nodes[batch_idx, right_nodes]
            ).to(dtype=compatibility.dtype)
        denom = pair_weight.sum(dim=-1).clamp_min(1)
        return (compatibility.log() * pair_weight).sum(dim=-1) / denom

    @torch.no_grad()
    def sample_precedence_reranked_beam_from_node_repr(
        self,
        raw_node_repr: Tensor,
        num_samples: int = 1,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.precedence_head is None:
            raise RuntimeError(
                "Precedence beam reranking requires a trained precedence head."
            )

        priority = None
        node_repr = raw_node_repr
        relation_logits = self.precedence_head(raw_node_repr)
        valid_nodes = None
        if mask is not None:
            valid_nodes = mask[:, -1, :] > -1e20

        if hasattr(self, "_p_sample_beam_search_with_priority"):
            batch_size, num_nodes, d_model = raw_node_repr.shape
            priority = self._sample_priorities(
                batch_size=num_samples * batch_size,
                num_nodes=num_nodes,
                device=raw_node_repr.device,
                dtype=raw_node_repr.dtype,
            )
            node_repr = (
                raw_node_repr.unsqueeze(0)
                .expand(num_samples, batch_size, num_nodes, d_model)
                .reshape(num_samples * batch_size, num_nodes, d_model)
            )
            relation_logits = (
                relation_logits.unsqueeze(0)
                .expand(num_samples, *relation_logits.shape)
                .reshape(num_samples * batch_size, num_nodes, num_nodes, 3)
            )
            if valid_nodes is not None:
                valid_nodes = (
                    valid_nodes.unsqueeze(0)
                    .expand(num_samples, *valid_nodes.shape)
                    .reshape(num_samples * batch_size, num_nodes)
                )
            candidates, diffusion_log_probs = self._p_sample_beam_search_with_priority(
                node_repr,
                priority_start=priority,
                return_candidates=True,
            )
        else:
            candidates, diffusion_log_probs = self.diffusion_utils.p_sample_beam_search(
                node_repr,
                self.reverse_model,
                return_candidates=True,
            )

        precedence_scores = self._precedence_candidate_scores(
            relation_logits,
            candidates,
            valid_nodes=valid_nodes,
        )
        num_decisions = max(
            (len(self.diffusion_utils.reverse_steps) - 1) * (candidates.size(-1) - 1),
            1,
        )
        diffusion_scores = diffusion_log_probs.float() / num_decisions
        combined_scores = (
            diffusion_scores
            + self.topo_precedence_rerank_beta * precedence_scores
        )
        best_idx = combined_scores.argmax(dim=-1)
        batch_idx = torch.arange(candidates.size(0), device=candidates.device)
        best_orders = candidates[batch_idx, best_idx]
        return best_orders, priority

    def sample(
        self,
        target_data: Tensor,
        num_samples: int = 1,
        mask: Optional[Tensor] = None,
    ):
        if (
            self.precedence_head is None
            or self.topo_precedence_rerank_beta == 0
        ):
            return super().sample(
                target_data,
                num_samples=num_samples,
                mask=mask,
            )

        raw_node_repr = self._encode_raw_data(target_data, mask=mask)
        was_training = self.reverse_model.training
        self.reverse_model.eval()
        try:
            orders, priority = self.sample_precedence_reranked_beam_from_node_repr(
                raw_node_repr,
                num_samples=num_samples,
                mask=mask,
            )
        finally:
            self.reverse_model.train(was_training)

        batch_size, num_nodes = raw_node_repr.shape[:2]
        if priority is None:
            orders = orders.unsqueeze(0).expand(num_samples, -1, -1)
            return orders, mask
        orders = orders.reshape(num_samples, batch_size, num_nodes)
        priority = priority.reshape(num_samples, batch_size, num_nodes)
        return orders, priority


class SourceLayerDecoder(nn.Module):
    """Predict the current source-node set from remaining node representations."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
        use_global_context: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        self.step_embedder = TimestepEmbedder(d_model, time_mlp=True)
        self.remaining_embed = nn.Linear(1, d_model, **factory_kwargs)
        self.use_global_context = use_global_context
        self.global_proj = nn.Linear(d_model, d_model, **factory_kwargs)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.logit_head = nn.Sequential(
            nn.Linear(d_model, d_model, **factory_kwargs),
            nn.GELU(),
            nn.Linear(d_model, 1, **factory_kwargs),
        )

    def forward(
        self,
        node_repr: Tensor,
        remaining_mask: Tensor,
        step: Tensor,
        valid_node_mask: Optional[Tensor] = None,
    ) -> Tensor:
        remaining_mask = remaining_mask.to(torch.bool)
        if valid_node_mask is None:
            valid_node_mask = remaining_mask
        else:
            valid_node_mask = valid_node_mask.to(torch.bool)

        step_emb = self.step_embedder(step.to(device=node_repr.device)).to(dtype=node_repr.dtype)
        remaining_emb = self.remaining_embed(
            remaining_mask.unsqueeze(-1).to(dtype=node_repr.dtype)
        )
        x = node_repr + step_emb.unsqueeze(1) + remaining_emb

        if self.use_global_context:
            weight = valid_node_mask.unsqueeze(-1).to(dtype=node_repr.dtype)
            denom = weight.sum(dim=1).clamp_min(1.0)
            global_repr = (node_repr * weight).sum(dim=1) / denom
            x = x + self.global_proj(global_repr).unsqueeze(1)

        x = self.encoder(x, src_key_padding_mask=~remaining_mask)
        logits = self.logit_head(x).squeeze(-1)
        return logits.masked_fill(~remaining_mask, -1e9)


class SourceLayerMixin:
    @staticmethod
    def _valid_nodes_from_decoder_mask(
        mask: Optional[Tensor],
        batch_size: int,
        num_nodes: int,
        device,
    ) -> Tensor:
        if mask is None:
            return torch.ones((batch_size, num_nodes), dtype=torch.bool, device=device)
        return mask[:, -1, :] > -1e20

    @staticmethod
    def _source_layers_from_graph(
        graph: Tensor,
        valid_nodes: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_nodes, _ = graph.shape
        source_targets = torch.zeros(
            (batch_size, num_nodes, num_nodes),
            dtype=torch.bool,
            device=graph.device,
        )
        remaining_masks = torch.zeros_like(source_targets)
        layer_ids = torch.full(
            (batch_size, num_nodes),
            fill_value=-1,
            dtype=torch.long,
            device=graph.device,
        )
        layer_counts = torch.zeros((batch_size,), dtype=torch.long, device=graph.device)

        for b in range(batch_size):
            remaining = valid_nodes[b].clone()
            for step in range(num_nodes):
                if not remaining.any():
                    break
                active_edges = (graph[b] > 0.5) & remaining.unsqueeze(1) & remaining.unsqueeze(0)
                indegree = active_edges.to(torch.long).sum(dim=0)
                source = remaining & (indegree == 0)
                if not source.any():
                    raise ValueError(
                        "Cannot build source-layer labels: target graph contains a cycle "
                        f"in batch item {b}."
                    )
                remaining_masks[b, step] = remaining
                source_targets[b, step] = source
                layer_ids[b, source] = step
                layer_counts[b] += 1
                remaining = remaining & ~source

        return source_targets, remaining_masks, layer_ids, layer_counts

    def _source_layer_loss_per_batch(
        self,
        node_repr: Tensor,
        graph: Tensor,
        mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_nodes = graph.shape[:2]
        valid_nodes = self._valid_nodes_from_decoder_mask(
            mask,
            batch_size=batch_size,
            num_nodes=num_nodes,
            device=graph.device,
        )
        source_targets, remaining_masks, layer_ids, layer_counts = self._source_layers_from_graph(
            graph=graph,
            valid_nodes=valid_nodes,
        )
        loss_sum = torch.zeros((batch_size,), dtype=node_repr.dtype, device=node_repr.device)
        pos_weight = torch.tensor(
            float(self.source_pos_weight),
            dtype=torch.float32,
            device=node_repr.device,
        )

        for step in range(num_nodes):
            active_batch = step < layer_counts
            if not active_batch.any():
                continue
            active_idx = torch.nonzero(active_batch, as_tuple=False).flatten()
            step_tensor = torch.full(
                (active_idx.numel(),),
                step,
                dtype=torch.long,
                device=node_repr.device,
            )
            remaining = remaining_masks[active_idx, step]
            logits = self.source_layer_decoder(
                node_repr[active_idx],
                remaining_mask=remaining,
                step=step_tensor,
                valid_node_mask=valid_nodes[active_idx],
            )
            target = source_targets[active_idx, step].to(dtype=logits.dtype)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(),
                target.float(),
                reduction="none",
                pos_weight=pos_weight,
            )
            weight = remaining.to(dtype=loss.dtype)
            denom = weight.sum(dim=1).clamp_min(1.0)
            loss_sum[active_idx] += (loss * weight).sum(dim=1).to(loss_sum.dtype) / denom.to(loss_sum.dtype)

        source_loss = loss_sum / layer_counts.clamp_min(1).to(dtype=loss_sum.dtype)
        return source_loss, layer_ids, source_targets, layer_counts

    def _decode_source_layers_from_node_repr(
        self,
        node_repr: Tensor,
        valid_nodes: Tensor,
        source_threshold: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        threshold = self.source_threshold if source_threshold is None else source_threshold
        batch_size, num_nodes = valid_nodes.shape
        remaining = valid_nodes.clone()
        layer_ids = torch.full(
            (batch_size, num_nodes),
            fill_value=-1,
            dtype=torch.long,
            device=node_repr.device,
        )
        layer_counts = torch.zeros((batch_size,), dtype=torch.long, device=node_repr.device)
        last_logits = torch.full(
            (batch_size, num_nodes),
            fill_value=-1e9,
            dtype=node_repr.dtype,
            device=node_repr.device,
        )

        for step in range(num_nodes):
            active_batch = remaining.any(dim=1)
            if not active_batch.any():
                break
            active_idx = torch.nonzero(active_batch, as_tuple=False).flatten()
            step_tensor = torch.full(
                (active_idx.numel(),),
                step,
                dtype=torch.long,
                device=node_repr.device,
            )
            logits = self.source_layer_decoder(
                node_repr[active_idx],
                remaining_mask=remaining[active_idx],
                step=step_tensor,
                valid_node_mask=valid_nodes[active_idx],
            )
            probs = torch.sigmoid(logits.float())
            chosen = (probs > threshold) & remaining[active_idx]
            empty = ~chosen.any(dim=1)
            if empty.any():
                fallback_logits = logits.masked_fill(~remaining[active_idx], -1e9)
                fallback = fallback_logits.argmax(dim=1)
                chosen[empty] = False
                chosen[empty, fallback[empty]] = True

            batch_ids = active_idx.unsqueeze(1).expand_as(chosen)
            node_ids = torch.arange(num_nodes, device=node_repr.device).view(1, -1).expand_as(chosen)
            layer_ids[batch_ids[chosen], node_ids[chosen]] = step
            layer_counts[active_idx] += 1
            remaining[active_idx] = remaining[active_idx] & ~chosen
            last_logits[active_idx] = logits

        return layer_ids, layer_counts, last_logits

    @staticmethod
    def _orders_from_layer_ids(layer_ids: Tensor, valid_nodes: Tensor) -> Tensor:
        batch_size, num_nodes = layer_ids.shape
        node_ids = torch.arange(num_nodes, device=layer_ids.device)
        orders = torch.empty((batch_size, num_nodes), dtype=torch.long, device=layer_ids.device)
        for b in range(batch_size):
            valid_order = node_ids[valid_nodes[b]][
                torch.argsort(layer_ids[b, valid_nodes[b]] * num_nodes + node_ids[valid_nodes[b]])
            ]
            invalid_order = node_ids[~valid_nodes[b]]
            orders[b] = torch.cat([valid_order, invalid_order], dim=0)
        return orders

    @staticmethod
    def _orient_skeleton_by_layer_ids(
        skeleton_logits: Tensor,
        layer_ids: Tensor,
        valid_nodes: Tensor,
        threshold: float,
    ) -> Tensor:
        probs = torch.sigmoid(skeleton_logits.float())
        probs = (probs + probs.transpose(1, 2)) / 2.0
        batch_size, num_nodes, _ = probs.shape
        pred = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.float32, device=probs.device)
        upper = torch.triu(
            torch.ones((num_nodes, num_nodes), dtype=torch.bool, device=probs.device),
            diagonal=1,
        )
        skeleton_edges = (probs > threshold) & upper.unsqueeze(0)
        valid_pairs = valid_nodes.unsqueeze(1) & valid_nodes.unsqueeze(2)
        skeleton_edges = skeleton_edges & valid_pairs
        i_before_j = layer_ids.unsqueeze(2) < layer_ids.unsqueeze(1)
        j_before_i = layer_ids.unsqueeze(1) < layer_ids.unsqueeze(2)
        pred = pred.masked_fill(skeleton_edges & i_before_j, 1.0)
        pred = pred.masked_fill((skeleton_edges & j_before_i).transpose(1, 2), 1.0)
        return pred

    @staticmethod
    def _is_acyclic_batch(adj: Tensor, valid_nodes: Tensor) -> Tensor:
        batch_size, num_nodes, _ = adj.shape
        result = torch.ones((batch_size,), dtype=torch.bool, device=adj.device)
        for b in range(batch_size):
            remaining = valid_nodes[b].clone()
            active_adj = (adj[b] > 0.5) & remaining.unsqueeze(1) & remaining.unsqueeze(0)
            for _ in range(num_nodes):
                if not remaining.any():
                    break
                indegree = active_adj.to(torch.long).sum(dim=0)
                source = remaining & (indegree == 0)
                if not source.any():
                    result[b] = False
                    break
                active_adj[source, :] = False
                active_adj[:, source] = False
                remaining[source] = False
        return result

    @staticmethod
    def _binary_precision_recall(
        target: Tensor,
        pred: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        target = target.to(torch.bool) & mask
        pred = pred.to(torch.bool) & mask
        tp = (target & pred).flatten(1).sum(dim=1).float()
        fp = (~target & pred & mask).flatten(1).sum(dim=1).float()
        fn = (target & ~pred & mask).flatten(1).sum(dim=1).float()
        precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
        recall = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp))
        f1 = torch.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            torch.zeros_like(precision),
        )
        return precision, recall, f1

    def evaluate_batch(
        self,
        target_data: Tensor,
        graph: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        raw_node_repr = self._encode_raw_data(target_data, mask=mask)
        batch_size, num_nodes = graph.shape[:2]
        valid_nodes = self._valid_nodes_from_decoder_mask(
            mask,
            batch_size=batch_size,
            num_nodes=num_nodes,
            device=graph.device,
        )
        true_source_loss, true_layer_ids, _, true_layer_counts = self._source_layer_loss_per_batch(
            raw_node_repr,
            graph=graph,
            mask=mask,
        )
        pred_layer_ids, pred_layer_counts, _ = self._decode_source_layers_from_node_repr(
            raw_node_repr,
            valid_nodes=valid_nodes,
        )
        skeleton_logits = self._skeleton_logits_from_node_repr(raw_node_repr, mask=mask)
        skeleton_loss = self._skeleton_loss_per_batch(skeleton_logits, graph=graph, mask=mask)
        pred_adj = self._orient_skeleton_by_layer_ids(
            skeleton_logits,
            layer_ids=pred_layer_ids,
            valid_nodes=valid_nodes,
            threshold=self.skeleton_threshold,
        )

        node_layer_accuracy = ((pred_layer_ids == true_layer_ids) | ~valid_nodes).float()
        node_layer_accuracy = (node_layer_accuracy * valid_nodes.float()).sum(dim=1) / valid_nodes.float().sum(dim=1).clamp_min(1.0)
        full_layering_accuracy = ((pred_layer_ids == true_layer_ids) | ~valid_nodes).all(dim=1).float()

        max_layers = torch.maximum(true_layer_counts, pred_layer_counts).max().item()
        source_precision_sum = torch.zeros((batch_size,), dtype=torch.float32, device=graph.device)
        source_recall_sum = torch.zeros_like(source_precision_sum)
        source_f1_sum = torch.zeros_like(source_precision_sum)
        source_metric_count = torch.zeros_like(source_precision_sum)
        for step in range(int(max_layers)):
            active_layer = (step < true_layer_counts) | (step < pred_layer_counts)
            true_set = (true_layer_ids == step) & valid_nodes
            pred_set = (pred_layer_ids == step) & valid_nodes
            tp = (true_set & pred_set).sum(dim=1).float()
            fp = (~true_set & pred_set & valid_nodes).sum(dim=1).float()
            fn = (true_set & ~pred_set & valid_nodes).sum(dim=1).float()
            precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
            recall = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp))
            f1 = torch.where(
                precision + recall > 0,
                2 * precision * recall / (precision + recall),
                torch.zeros_like(precision),
            )
            active_weight = active_layer.float()
            source_precision_sum += precision * active_weight
            source_recall_sum += recall * active_weight
            source_f1_sum += f1 * active_weight
            source_metric_count += active_weight
        source_metric_count = source_metric_count.clamp_min(1.0)
        source_precision = source_precision_sum / source_metric_count
        source_recall = source_recall_sum / source_metric_count
        source_f1 = source_f1_sum / source_metric_count

        true_edge_mask = valid_nodes.unsqueeze(1) & valid_nodes.unsqueeze(2)
        eye = torch.eye(num_nodes, dtype=torch.bool, device=graph.device).unsqueeze(0)
        true_edge_mask = true_edge_mask & ~eye
        final_precision, final_recall, _ = self._binary_precision_recall(
            target=graph > 0.5,
            pred=pred_adj > 0.5,
            mask=true_edge_mask,
        )
        final_shd = ((graph > 0.5) ^ (pred_adj > 0.5)).to(torch.float32)
        final_shd = (final_shd * true_edge_mask.float()).flatten(1).sum(dim=1)

        upper = torch.triu(
            torch.ones((num_nodes, num_nodes), dtype=torch.bool, device=graph.device),
            diagonal=1,
        ).unsqueeze(0)
        pair_mask = upper & valid_nodes.unsqueeze(1) & valid_nodes.unsqueeze(2)
        true_skeleton = ((graph > 0.5) | (graph.transpose(1, 2) > 0.5))
        pred_skeleton = torch.sigmoid(skeleton_logits.float()) > self.skeleton_threshold
        skeleton_precision, skeleton_recall, _ = self._binary_precision_recall(
            target=true_skeleton,
            pred=pred_skeleton,
            mask=pair_mask,
        )

        topo_valid = torch.ones((batch_size,), dtype=torch.bool, device=graph.device)
        edge_mask = (graph > 0.5) & true_edge_mask
        for b in range(batch_size):
            if edge_mask[b].any():
                src, dst = torch.nonzero(edge_mask[b], as_tuple=True)
                topo_valid[b] = (pred_layer_ids[b, src] < pred_layer_ids[b, dst]).all()

        return {
            "source_layer_loss": true_source_loss.detach(),
            "node_layer_accuracy": node_layer_accuracy.detach(),
            "full_layering_accuracy": full_layering_accuracy.detach(),
            "source_layer_precision": source_precision.detach(),
            "source_layer_recall": source_recall.detach(),
            "source_layer_f1": source_f1.detach(),
            "predicted_layer_count": pred_layer_counts.float().detach(),
            "topo_order_validity": topo_valid.float().detach(),
            "skeleton_loss": skeleton_loss.detach(),
            "skeleton_precision": skeleton_precision.detach(),
            "skeleton_recall": skeleton_recall.detach(),
            "final_adjacency_acyclic": self._is_acyclic_batch(pred_adj, valid_nodes).float().detach(),
            "final_shd": final_shd.detach(),
            "final_edge_precision": final_precision.detach(),
            "final_edge_recall": final_recall.detach(),
        }


class CausalSkeletonDecoder(BakStyleSkeletonMixin, CausalTNPEncoder):
    """Skeleton-only decoder using the causal encoder and a bak-style L head."""

    def __init__(
        self,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers_encoder,
        num_layers_decoder,
        num_nodes,
        n_perm_samples=None,
        sinkhorn_iter=None,
        use_positional_encoding=False,
        skeleton_loss_weight: float = 1.0,
        skeleton_decoder_layers: int = 2,
        device=None,
        dtype=None,
        mlp_use_bias: bool = False,
        **kwargs,
    ):
        super(CausalSkeletonDecoder, self).__init__(
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
            mlp_use_bias=mlp_use_bias,
        )
        self.num_nodes = num_nodes
        self._init_skeleton_head(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers_skeleton=skeleton_decoder_layers,
            dropout=dropout,
            skeleton_loss_weight=skeleton_loss_weight,
            order_loss_weight=0.0,
            device=device,
            dtype=dtype,
        )

    def _encode_raw_data(self, target_data: Tensor, mask: Optional[Tensor]) -> Tensor:
        if target_data.dim() == 3:
            target_data = target_data.unsqueeze(-1)
        return self.encode(target_data=target_data, mask=mask).squeeze(2)

    def forward(
        self,
        target_data: Tensor,
        graph: Optional[Tensor],
        mask: Optional[Tensor] = None,
        is_training: bool = True,
    ):
        skeleton_logits = self._skeleton_logits_from_data(target_data, mask=mask)
        output = {"skeleton_logits": skeleton_logits}
        if graph is not None:
            skeleton_loss = self._skeleton_loss_per_batch(
                skeleton_logits,
                graph=graph,
                mask=mask,
            )
            output["loss"] = self.skeleton_loss_weight * skeleton_loss
            output["skeleton_loss"] = skeleton_loss
        return output

    def calculate_loss(self, output, target):
        if not isinstance(output, dict) or "loss" not in output:
            raise ValueError("CausalSkeletonDecoder expects forward() to return a loss dict.")
        return output["loss"]

    def sample(self, target_data: Tensor, num_samples: int = 1, mask: Optional[Tensor] = None):
        logits = self._skeleton_logits_from_data(target_data, mask=mask)
        probs = torch.sigmoid(logits)
        eye = torch.eye(probs.size(-1), device=probs.device, dtype=probs.dtype)
        probs = probs * (1 - eye)
        upper_probs = torch.triu(probs, diagonal=1)
        upper_samples = torch.distributions.Bernoulli(probs=upper_probs).sample((num_samples,))
        samples = upper_samples + upper_samples.transpose(-1, -2)
        return samples, probs


class CausalEmbeddingReverseDiffusion(nn.Module):
    """
    SymmetricDiffusers-style reverse model for already embedded causal nodes.

    Args:
        src: current noisy permutation(s), shape [N, T, B, V] during training or
            [B, beam, V] during sampling.
        time: diffusion timestep(s), shape [T, 1] during training or [B] at eval.
        x_start: embedded nodes in the input order, shape [B, V, D].

    Returns:
        Generalized-PL scores with shape [..., V, V].
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_out_adjust = "square"
        self.time_embd = TimestepEmbedder(d_model, time_mlp=True)
        self.encoder_layers = EncoderLayers(
            dataset="sort-MNIST",
            d_model=d_model,
            nhead=nhead,
            d_hid=dim_feedforward,
            nlayers=num_layers,
            dropout=dropout,
            d_out_adjust=self.d_out_adjust,
            encoder="original",
        )

    @staticmethod
    def _permute_embd(perm_list: Tensor, x: Tensor) -> Tensor:
        x, perm_list = torch.broadcast_tensors(x, perm_list.unsqueeze(-1))
        return torch.gather(x, -2, perm_list)

    def training_patch_embd(self, src: Tensor, x_start: Tensor) -> Tensor:
        src = self._permute_embd(src, x_start)
        return torch.flatten(src, end_dim=-3)

    def eval_patch_embd(self, src: Tensor, x_start: Tensor) -> Tensor:
        x_start = x_start.unsqueeze(-3)
        src = self._permute_embd(src, x_start)
        return torch.flatten(src, end_dim=-3)

    def forward(self, src: Tensor, time: Tensor, x_start: Tensor) -> Tensor:
        batch_shape = src.shape[:-1]
        num_nodes = src.size(-1)

        if src.dim() == 4:
            time = time.expand(batch_shape)
            src = self.training_patch_embd(src, x_start)
        else:
            time = time.unsqueeze(-1).expand(batch_shape)
            src = self.eval_patch_embd(src, x_start)

        time = time.flatten()
        time_embd = self.time_embd(time).to(dtype=src.dtype)

        out = self.encoder_layers(src, time_embd)
        row, col = torch.split(out, [num_nodes, num_nodes], dim=-2)
        scores = torch.matmul(row, col.transpose(-1, -2))
        return scores.unflatten(0, batch_shape)


class PriorityCausalEmbeddingReverseDiffusion(CausalEmbeddingReverseDiffusion):
    """
    Reverse denoiser with exogenous priority conditioning.

    The causal encoder representation is unchanged. At each reverse step,
    priority is gathered with the same current noisy permutation as the node
    embeddings, embedded, concatenated to the node embeddings, and fused back to
    d_model before the usual generalized-PL score network.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
        priority_scale_init: float = -2.0,
        priority_emb_dim: int = 16,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout=dropout,
        )
        # Kept in the signature for old configs/commands. This is now used as
        # the initial log-gate for the learned priority column residual, not as
        # a hand-written score bias.
        self.priority_emb_dim = priority_emb_dim
        self.priority_embedder = nn.Sequential(
            nn.Linear(2, priority_emb_dim),
            nn.GELU(),
            nn.Linear(priority_emb_dim, priority_emb_dim),
        )
        self.priority_fuse = nn.Sequential(
            nn.Linear(d_model + priority_emb_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.priority_col_fuse = nn.Linear(d_model + priority_emb_dim, d_model)
        self.priority_col_gate = nn.Parameter(torch.tensor(float(priority_scale_init)))

    @staticmethod
    def _permute_priority(perm_list: Tensor, priority: Tensor) -> Tensor:
        priority = priority.unsqueeze(-1)
        priority, perm_list = torch.broadcast_tensors(priority, perm_list.unsqueeze(-1))
        return torch.gather(priority, -2, perm_list).squeeze(-1)

    def training_patch_priority(self, src: Tensor, priority_start: Tensor) -> Tensor:
        return self._permute_priority(src, priority_start)

    def eval_patch_priority(self, src: Tensor, priority_start: Tensor) -> Tensor:
        return self._permute_priority(src, priority_start.unsqueeze(-2))

    @staticmethod
    def _priority_features(priority: Tensor) -> Tensor:
        priority = priority.float()
        rank = priority.argsort(dim=-1).argsort(dim=-1).to(dtype=priority.dtype)
        denom = max(priority.size(-1) - 1, 1)
        rank = rank / denom
        return torch.stack([priority, rank], dim=-1)

    def _fuse_priority(self, src: Tensor, priority_noisy: Tensor) -> Tuple[Tensor, Tensor]:
        priority_noisy = torch.flatten(priority_noisy, end_dim=-2)
        priority_features = self._priority_features(priority_noisy).to(
            device=src.device,
            dtype=src.dtype,
        )
        priority_emb = self.priority_embedder(priority_features)
        src = self.priority_fuse(torch.cat([src, priority_emb], dim=-1))
        return src, priority_emb

    def forward(
        self,
        src: Tensor,
        time: Tensor,
        x_start: Tensor,
        priority_start: Tensor,
    ) -> Tensor:
        batch_shape = src.shape[:-1]
        num_nodes = src.size(-1)

        if src.dim() == 4:
            priority_noisy = self.training_patch_priority(src, priority_start)
            time = time.expand(batch_shape)
            src = self.training_patch_embd(src, x_start)
        else:
            priority_noisy = self.eval_patch_priority(src, priority_start)
            time = time.unsqueeze(-1).expand(batch_shape)
            src = self.eval_patch_embd(src, x_start)

        src, priority_emb = self._fuse_priority(src, priority_noisy)
        time = time.flatten()
        time_embd = self.time_embd(time).to(dtype=src.dtype)

        out = self.encoder_layers(src, time_embd)
        row, col = torch.split(out, [num_nodes, num_nodes], dim=-2)
        col_delta = self.priority_col_fuse(torch.cat([col, priority_emb], dim=-1))
        priority_col_scale = torch.nn.functional.softplus(self.priority_col_gate).to(
            device=col.device,
            dtype=col.dtype,
        )
        col = col + priority_col_scale * col_delta
        scores = torch.matmul(row, col.transpose(-1, -2))
        return scores.unflatten(0, batch_shape)


class CausalTopoOrderDiffusion(CausalTNPEncoder):
    """
    Learn topological orders with SymmetricDiffusers' permutation diffusion loss.

    Training samples one valid topological order per DAG, reorders the raw
    variables into that order, encodes the ordered data with CausalTNPEncoder,
    and applies the original permutation diffusion likelihood to the resulting
    clean sequence.
    """

    def __init__(
        self,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers_encoder,
        num_layers_decoder,
        num_nodes,
        n_perm_samples,
        sinkhorn_iter,
        use_positional_encoding,
        topo_num_timesteps: int = 7,
        topo_sample_N: int = 1,
        topo_transition: str = "riffle",
        topo_reverse: str = "generalized_PL",
        topo_reverse_steps: Optional[list[int]] = None,
        topo_beam_size: int = 20,
        device=None,
        dtype=None,
        mlp_use_bias: bool = False,
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            num_layers=num_layers_encoder,
            emb_depth=emb_depth,
            use_positional_encoding=use_positional_encoding,
            num_nodes=num_nodes,
            dropout=dropout,
            device=device,
            dtype=dtype,
            mlp_use_bias=mlp_use_bias,
        )
        self.num_nodes = num_nodes
        self.topo_num_timesteps = topo_num_timesteps
        self.topo_sample_N = topo_sample_N
        self.topo_transition = topo_transition
        self.topo_reverse = topo_reverse
        self.topo_reverse_steps = [] if topo_reverse_steps is None else topo_reverse_steps

        self.reverse_model = CausalEmbeddingReverseDiffusion(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers_decoder,
            dropout=dropout,
        )
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        if factory_kwargs:
            self.reverse_model.to(**factory_kwargs)
        self.diffusion_utils = DiffusionUtils(
            num_timesteps=topo_num_timesteps,
            sample_N=topo_sample_N,
            transition=topo_transition,
            latent=False,
            reinforce_N=10,
            reinforce_ema_rate=0.995,
            entropy_reg_rate=0.05,
            reverse=topo_reverse,
            reverse_steps=self.topo_reverse_steps,
            loss="log_likelihood",
            beam_size={"PL": topo_beam_size, "time": topo_beam_size},
            perm_fix_first=False,
        )

    @staticmethod
    def _valid_nodes_from_mask(mask: Optional[Tensor], num_nodes: int, device) -> Tensor:
        if mask is None:
            return torch.ones((0, num_nodes), dtype=torch.bool, device=device)
        return mask[:, -1, :] > -1e20

    def _sample_batch_topological_orders(
        self,
        graph: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, num_nodes, _ = graph.shape
        if mask is None:
            valid_nodes = torch.ones((batch_size, num_nodes), dtype=torch.bool, device=graph.device)
        else:
            valid_nodes = mask[:, -1, :] > -1e20

        orders = torch.empty((batch_size, num_nodes), dtype=torch.long, device=graph.device)
        for b in range(batch_size):
            valid_idx = torch.nonzero(valid_nodes[b], as_tuple=False).flatten()
            invalid_idx = torch.nonzero(~valid_nodes[b], as_tuple=False).flatten()
            subgraph = graph[b][valid_idx][:, valid_idx]
            local_order = random_kahn_topological_sort(subgraph)
            ordered_valid = valid_idx[
                torch.tensor(local_order, dtype=torch.long, device=graph.device)
            ]
            orders[b] = torch.cat([ordered_valid, invalid_idx], dim=0)
        return orders

    @staticmethod
    def _reorder_nodes(x: Tensor, orders: Tensor) -> Tensor:
        if x.dim() == 3:
            gather_idx = orders.unsqueeze(1).expand(-1, x.size(1), -1)
            return torch.gather(x, 2, gather_idx)
        if x.dim() == 4:
            gather_idx = orders.unsqueeze(1).unsqueeze(-1).expand(-1, x.size(1), -1, x.size(-1))
            return torch.gather(x, 2, gather_idx)
        raise ValueError("Expected node data with shape [B, S, V] or [B, S, V, C].")

    @staticmethod
    def _reorder_node_repr(node_repr: Tensor, orders: Tensor) -> Tensor:
        if node_repr.dim() != 3:
            raise ValueError("Expected node representations with shape [B, V, D].")
        gather_idx = orders.unsqueeze(-1).expand(-1, -1, node_repr.size(-1))
        return torch.gather(node_repr, 1, gather_idx)

    @staticmethod
    def _reorder_mask(mask: Optional[Tensor], orders: Tensor) -> Optional[Tensor]:
        if mask is None:
            return None
        gather_idx = orders.unsqueeze(1).expand(-1, mask.size(1), -1)
        return torch.gather(mask, 2, gather_idx)

    def _training_losses_per_batch(self, x_start: Tensor) -> Tensor:
        device = x_start.device
        num_nodes = x_start.size(1)
        batch_size = x_start.size(0)

        identity_perm = torch.arange(num_nodes, device=device).expand(batch_size, -1)
        perm_seq = self.diffusion_utils.q_sample_seq(identity_perm)
        perm_seq = perm_seq[:, self.diffusion_utils.reverse_steps, ...]
        perm_seq_no_start = perm_seq[:, 1:, ...]
        perm_seq_no_end = perm_seq[:, :-1, ...]

        t = torch.tensor(self.diffusion_utils.reverse_steps[1:], device=device).unsqueeze(-1)
        scores = self.diffusion_utils.p_logits(
            self.reverse_model,
            perm_seq_no_start,
            t,
            x_start,
        )

        p_log_probs = self.diffusion_utils.p_log_cond_prob(
            scores.float(),
            perm_tm1=perm_seq_no_end,
            perm_t=perm_seq_no_start,
        )
        loss = -p_log_probs.mean(dim=1)
        return loss.mean(dim=0)

    def _encode_ordered_data(
        self,
        target_data: Tensor,
        graph: Tensor,
        mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        orders = self._sample_batch_topological_orders(graph, mask=mask)
        target_data = self._reorder_nodes(target_data, orders)
        mask = self._reorder_mask(mask, orders)
        if target_data.dim() == 3:
            target_data = target_data.unsqueeze(-1)
        node_repr = self.encode(target_data=target_data, mask=mask).squeeze(2)
        return node_repr, orders

    def _encode_raw_data(self, target_data: Tensor, mask: Optional[Tensor]) -> Tensor:
        if target_data.dim() == 3:
            target_data = target_data.unsqueeze(-1)
        return self.encode(target_data=target_data, mask=mask).squeeze(2)

    def forward(
        self,
        target_data: Tensor,
        graph: Optional[Tensor],
        mask: Optional[Tensor] = None,
        is_training: bool = True,
    ):
        if graph is None:
            node_repr = self._encode_raw_data(target_data, mask=mask)
            was_training = self.reverse_model.training
            self.reverse_model.eval()
            try:
                _, orders = self.diffusion_utils.p_sample_loop(
                    node_repr,
                    self.reverse_model,
                    deterministic=True,
                )
            finally:
                self.reverse_model.train(was_training)
            return {"orders": orders}

        node_repr, clean_orders = self._encode_ordered_data(target_data, graph=graph, mask=mask)
        loss = self._training_losses_per_batch(node_repr)
        return {
            "loss": loss,
            "clean_orders": clean_orders,
        }

    def calculate_loss(self, output, target):
        if not isinstance(output, dict) or "loss" not in output:
            raise ValueError("CausalTopoOrderDiffusion expects forward() to return a loss dict.")
        return output["loss"]

    def sample(self, target_data: Tensor, num_samples: int = 1, mask: Optional[Tensor] = None):
        node_repr = self._encode_raw_data(target_data, mask=mask)
        orders = []
        was_training = self.reverse_model.training
        self.reverse_model.eval()
        try:
            for _ in range(num_samples):
                _, order = self.diffusion_utils.p_sample_loop(
                    node_repr,
                    self.reverse_model,
                    deterministic=False,
                )
                orders.append(order)
        finally:
            self.reverse_model.train(was_training)
        return torch.stack(orders, dim=0), mask

    @staticmethod
    def order_edge_precedence_accuracy(orders: Tensor, graph: Tensor) -> Tensor:
        if orders.dim() == 3:
            orders = orders[0]
        positions = torch.empty_like(orders)
        arange = torch.arange(orders.size(1), device=orders.device).expand_as(orders)
        positions.scatter_(1, orders.long(), arange)
        parent_pos = positions.unsqueeze(2)
        child_pos = positions.unsqueeze(1)
        edge_mask = graph > 0.5
        correct = parent_pos < child_pos
        edge_count = edge_mask.flatten(1).sum(dim=1)
        denom = edge_count.clamp_min(1)
        accuracy = (correct & edge_mask).flatten(1).sum(dim=1).float() / denom.float()
        return torch.where(edge_count > 0, accuracy, torch.ones_like(accuracy))


class CausalOrderGumbelSinkhorn(CausalTNPEncoder):
    """Order-only Gumbel-Sinkhorn baseline with topological-order supervision.

    Observation sampling, normalization, and the causal encoder match
    ``CausalTopoOrderDiffusion``. For every label DAG, training samples one
    valid root-to-leaf topological order with the same randomized Kahn routine.
    Unlike diffusion, the one-step decoder must encode nodes in their raw order:
    feeding it label-ordered nodes would expose the identity target directly.
    The decoder predicts a full node-to-position assignment matrix and is
    trained through soft Gumbel-Sinkhorn samples; it does not predict graph
    edges or a skeleton.
    """

    def __init__(
        self,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers_encoder,
        num_layers_decoder,
        num_nodes,
        n_perm_samples,
        sinkhorn_iter,
        use_positional_encoding,
        gs_temperature: float = 1.0,
        gs_noise_factor: float = 1.0,
        gs_train_samples: int = 1,
        device=None,
        dtype=None,
        mlp_use_bias: bool = False,
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            num_layers=num_layers_encoder,
            emb_depth=emb_depth,
            use_positional_encoding=use_positional_encoding,
            num_nodes=num_nodes,
            dropout=dropout,
            device=device,
            dtype=dtype,
            mlp_use_bias=mlp_use_bias,
        )
        if gs_temperature <= 0:
            raise ValueError("gs_temperature must be positive.")
        if n_perm_samples <= 0:
            raise ValueError("n_perm_samples must be positive.")
        if sinkhorn_iter <= 0:
            raise ValueError("sinkhorn_iter must be positive.")
        if gs_train_samples <= 0:
            raise ValueError("gs_train_samples must be positive.")

        self.num_nodes = num_nodes
        self.n_perm_samples = n_perm_samples
        self.sinkhorn_iter = sinkhorn_iter
        self.gs_temperature = gs_temperature
        self.gs_noise_factor = gs_noise_factor
        self.gs_train_samples = gs_train_samples

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.order_decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers_decoder,
            norm=nn.LayerNorm(d_model, device=device, dtype=dtype),
        )
        self.node_projection = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.position_queries = nn.Parameter(
            torch.empty(num_nodes, d_model, device=device, dtype=dtype)
        )
        self.position_projection = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        nn.init.normal_(self.position_queries, mean=0.0, std=d_model ** -0.5)

    @staticmethod
    def _decoder_padding_mask(mask: Optional[Tensor]) -> Optional[Tensor]:
        if mask is None:
            return None
        return mask[:, -1, :] <= -1e20

    def _encode_raw_data(self, target_data: Tensor, mask: Optional[Tensor]) -> Tensor:
        if target_data.dim() == 3:
            target_data = target_data.unsqueeze(-1)
        return self.encode(target_data=target_data, mask=mask).squeeze(2)

    def _sample_batch_topological_orders(
        self,
        graph: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, num_nodes, _ = graph.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {num_nodes}.")
        if mask is None:
            valid_nodes = torch.ones(
                (batch_size, num_nodes),
                dtype=torch.bool,
                device=graph.device,
            )
        else:
            valid_nodes = mask[:, -1, :] > -1e20
        if not valid_nodes.all():
            raise ValueError("topo_gs_order currently requires a fixed, unpadded node count.")

        orders = torch.empty((batch_size, num_nodes), dtype=torch.long, device=graph.device)
        for b in range(batch_size):
            local_order = random_kahn_topological_sort(graph[b])
            orders[b] = torch.tensor(local_order, dtype=torch.long, device=graph.device)
        return orders

    def _order_logits_from_node_repr(
        self,
        node_repr: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        padding_mask = self._decoder_padding_mask(mask)
        if padding_mask is not None and padding_mask.any():
            raise ValueError("topo_gs_order currently requires a fixed, unpadded node count.")
        decoded = self.order_decoder(
            node_repr,
            src_key_padding_mask=padding_mask,
        )
        node_features = self.node_projection(decoded)
        position_features = self.position_projection(
            self.position_queries[: node_repr.size(1)]
        )
        return torch.einsum("bnd,md->bnm", node_features, position_features) / math.sqrt(
            node_features.size(-1)
        )

    @staticmethod
    def _orders_to_node_positions(orders: Tensor) -> Tensor:
        positions = torch.empty_like(orders)
        rank = torch.arange(orders.size(1), device=orders.device).expand_as(orders)
        positions.scatter_(1, orders.long(), rank)
        return positions

    def _order_loss_per_batch(self, logits: Tensor, clean_orders: Tensor) -> Tensor:
        soft_permutations, _ = sample_permutation(
            log_alpha=logits,
            temp=self.gs_temperature,
            noise_factor=self.gs_noise_factor,
            n_samples=self.gs_train_samples,
            n_iters=self.sinkhorn_iter,
            squeeze=False,
            hard=False,
            device=logits.device,
        )
        target_positions = self._orders_to_node_positions(clean_orders)
        target = F.one_hot(
            target_positions,
            num_classes=logits.size(-1),
        ).to(dtype=soft_permutations.dtype)
        nll = -(
            target.unsqueeze(1)
            * soft_permutations.float().clamp_min(1e-12).log()
        ).sum(dim=(-1, -2)) / logits.size(-1)
        return nll.mean(dim=1)

    @staticmethod
    def _hard_assignments_to_orders(assignments: Tensor) -> Tensor:
        node_positions = assignments.argmax(dim=-1)
        return node_positions.argsort(dim=-1)

    def sample_orders_from_node_repr(
        self,
        node_repr: Tensor,
        num_samples: int = 1,
        mask: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> Tensor:
        logits = self._order_logits_from_node_repr(node_repr, mask=mask)
        assignments, _ = sample_permutation(
            log_alpha=logits,
            temp=self.gs_temperature,
            noise_factor=0.0 if deterministic else self.gs_noise_factor,
            n_samples=num_samples,
            n_iters=self.sinkhorn_iter,
            squeeze=False,
            hard=True,
            device=logits.device,
        )
        orders = self._hard_assignments_to_orders(assignments)
        return orders.transpose(0, 1).contiguous()

    def forward(
        self,
        target_data: Tensor,
        graph: Optional[Tensor],
        mask: Optional[Tensor] = None,
        is_training: bool = True,
    ):
        node_repr = self._encode_raw_data(target_data, mask=mask)
        if graph is None:
            orders = self.sample_orders_from_node_repr(
                node_repr,
                num_samples=1,
                mask=mask,
                deterministic=True,
            )
            return {"orders": orders[0]}

        clean_orders = self._sample_batch_topological_orders(graph, mask=mask)
        logits = self._order_logits_from_node_repr(node_repr, mask=mask)
        order_loss = self._order_loss_per_batch(logits, clean_orders)
        return {
            "loss": order_loss,
            "order_loss": order_loss,
            "clean_orders": clean_orders,
            "order_logits": logits,
        }

    def calculate_loss(self, output, target):
        if not isinstance(output, dict) or "loss" not in output:
            raise ValueError("CausalOrderGumbelSinkhorn expects forward() to return a loss dict.")
        return output["loss"]

    def sample(
        self,
        target_data: Tensor,
        num_samples: int = 1,
        mask: Optional[Tensor] = None,
    ):
        node_repr = self._encode_raw_data(target_data, mask=mask)
        orders = self.sample_orders_from_node_repr(
            node_repr,
            num_samples=num_samples,
            mask=mask,
            deterministic=False,
        )
        return orders, mask

    order_edge_precedence_accuracy = staticmethod(
        CausalTopoOrderDiffusion.order_edge_precedence_accuracy
    )


class CausalPriorityTopoOrderDiffusion(CausalTopoOrderDiffusion):
    """
    Topological-order diffusion conditioned on exogenous node priorities.

    A priority vector u ~ Uniform(0, 1)^D makes the training order unique via
    priority Kahn sorting. During denoising, priorities are gathered with the
    same current noisy permutation as node embeddings and bias generalized-PL
    candidate scores toward smaller priorities.
    """

    def __init__(
        self,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers_encoder,
        num_layers_decoder,
        num_nodes,
        n_perm_samples,
        sinkhorn_iter,
        use_positional_encoding,
        topo_num_timesteps: int = 7,
        topo_sample_N: int = 1,
        topo_transition: str = "riffle",
        topo_reverse: str = "generalized_PL",
        topo_reverse_steps: Optional[list[int]] = None,
        topo_beam_size: int = 20,
        topo_priority_scale_init: float = -2.0,
        topo_priority_mode: str = "random",
        device=None,
        dtype=None,
        mlp_use_bias: bool = False,
        **kwargs,
    ):
        if topo_reverse != "generalized_PL":
            raise ValueError("CausalPriorityTopoOrderDiffusion currently supports generalized_PL only.")
        if topo_priority_mode not in {"random", "fixed_node_order"}:
            raise ValueError(
                "topo_priority_mode must be 'random' or 'fixed_node_order', "
                f"got {topo_priority_mode!r}."
            )
        super().__init__(
            d_model=d_model,
            emb_depth=emb_depth,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            dropout=dropout,
            num_layers_encoder=num_layers_encoder,
            num_layers_decoder=num_layers_decoder,
            num_nodes=num_nodes,
            n_perm_samples=n_perm_samples,
            sinkhorn_iter=sinkhorn_iter,
            use_positional_encoding=use_positional_encoding,
            topo_num_timesteps=topo_num_timesteps,
            topo_sample_N=topo_sample_N,
            topo_transition=topo_transition,
            topo_reverse=topo_reverse,
            topo_reverse_steps=topo_reverse_steps,
            topo_beam_size=topo_beam_size,
            device=device,
            dtype=dtype,
            mlp_use_bias=mlp_use_bias,
            **kwargs,
        )
        self.topo_priority_mode = topo_priority_mode
        self.reverse_model = PriorityCausalEmbeddingReverseDiffusion(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers_decoder,
            dropout=dropout,
            priority_scale_init=topo_priority_scale_init,
        )
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        if factory_kwargs:
            self.reverse_model.to(**factory_kwargs)

    def _sample_priorities(self, batch_size: int, num_nodes: int, device, dtype) -> Tensor:
        # Keep exogenous priorities in fp32 even when the denoiser uses bf16.
        # This avoids unnecessary ties from low-precision priority sampling.
        if self.topo_priority_mode == "random":
            return torch.rand((batch_size, num_nodes), device=device, dtype=torch.float32)
        if num_nodes <= 1:
            base = torch.zeros((num_nodes,), device=device, dtype=torch.float32)
        else:
            base = torch.arange(num_nodes, device=device, dtype=torch.float32) / float(num_nodes - 1)
        return base.unsqueeze(0).expand(batch_size, -1)

    def _sample_batch_priority_topological_orders(
        self,
        graph: Tensor,
        priority: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, num_nodes, _ = graph.shape
        if mask is None:
            valid_nodes = torch.ones((batch_size, num_nodes), dtype=torch.bool, device=graph.device)
        else:
            valid_nodes = mask[:, -1, :] > -1e20

        return self._batched_priority_kahn_topological_orders(
            graph=graph,
            priority=priority,
            valid_nodes=valid_nodes,
        )

    @staticmethod
    def _batched_priority_kahn_topological_orders(
        graph: Tensor,
        priority: Tensor,
        valid_nodes: Tensor,
    ) -> Tensor:
        """
        Batched priority Kahn topological sort.

        This is equivalent to priority_kahn_topological_sort on each valid
        subgraph: among currently available source nodes, pick the node with
        the smallest priority, breaking exact ties by node index. Invalid padded
        nodes are appended at the end in ascending node order.
        """
        batch_size, num_nodes, _ = graph.shape
        device = graph.device
        node_ids = torch.arange(num_nodes, device=device)
        batch_ids = torch.arange(batch_size, device=device)

        valid_nodes = valid_nodes.to(torch.bool)
        active_edges = (graph > 0.5) & valid_nodes.unsqueeze(2) & valid_nodes.unsqueeze(1)
        indegree = active_edges.to(torch.long).sum(dim=1)
        selected = ~valid_nodes.clone()
        valid_counts = valid_nodes.to(torch.long).sum(dim=1)

        orders = torch.empty((batch_size, num_nodes), dtype=torch.long, device=device)
        for step in range(num_nodes):
            active_batch = step < valid_counts
            available = valid_nodes & ~selected & (indegree == 0)
            score = priority.masked_fill(~available, float("inf"))
            choice = score.argmin(dim=1)

            active_ids = batch_ids[active_batch]
            active_choice = choice[active_batch]
            orders[active_ids, step] = active_choice
            selected[active_ids, active_choice] = True
            indegree[active_ids] -= active_edges[active_ids, active_choice].to(torch.long)

        invalid_scores = torch.where(
            ~valid_nodes,
            node_ids.view(1, num_nodes).expand(batch_size, -1),
            torch.full((batch_size, num_nodes), num_nodes, dtype=torch.long, device=device),
        )
        invalid_sorted = invalid_scores.sort(dim=1).values
        pos = node_ids.view(1, num_nodes).expand(batch_size, -1)
        invalid_mask = pos >= valid_counts.unsqueeze(1)
        invalid_pos = (pos - valid_counts.unsqueeze(1)).clamp_min(0)
        orders[invalid_mask] = invalid_sorted.gather(1, invalid_pos)[invalid_mask]
        return orders

    @staticmethod
    def _reorder_priority(priority: Tensor, orders: Tensor) -> Tensor:
        return torch.gather(priority, 1, orders.long())

    def _training_losses_per_batch_with_priority(
        self,
        x_start: Tensor,
        priority_start: Tensor,
    ) -> Tensor:
        device = x_start.device
        num_nodes = x_start.size(1)
        batch_size = x_start.size(0)

        identity_perm = torch.arange(num_nodes, device=device).expand(batch_size, -1)
        perm_seq = self.diffusion_utils.q_sample_seq(identity_perm)
        perm_seq = perm_seq[:, self.diffusion_utils.reverse_steps, ...]
        perm_seq_no_start = perm_seq[:, 1:, ...]
        perm_seq_no_end = perm_seq[:, :-1, ...]

        t = torch.tensor(self.diffusion_utils.reverse_steps[1:], device=device).unsqueeze(-1)
        scores = self.reverse_model(
            perm_seq_no_start,
            t,
            x_start,
            priority_start,
        )

        p_log_probs = self.diffusion_utils.p_log_cond_prob(
            scores.float(),
            perm_tm1=perm_seq_no_end,
            perm_t=perm_seq_no_start,
        )
        loss = -p_log_probs.mean(dim=1)
        return loss.mean(dim=0)

    def _encode_priority_ordered_data(
        self,
        target_data: Tensor,
        graph: Tensor,
        mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, _, num_nodes = target_data.shape[:3]
        priority = self._sample_priorities(
            batch_size=batch_size,
            num_nodes=num_nodes,
            device=target_data.device,
            dtype=target_data.dtype,
        )
        orders = self._sample_batch_priority_topological_orders(
            graph,
            priority=priority,
            mask=mask,
        )
        target_data = self._reorder_nodes(target_data, orders)
        priority_ordered = self._reorder_priority(priority, orders)
        mask = self._reorder_mask(mask, orders)
        if target_data.dim() == 3:
            target_data = target_data.unsqueeze(-1)
        node_repr = self.encode(target_data=target_data, mask=mask).squeeze(2)
        return node_repr, priority_ordered, orders, priority

    def _p_sample_loop_with_priority(
        self,
        x_start: Tensor,
        priority_start: Tensor,
        deterministic: bool,
    ) -> Tuple[Tensor, Tensor]:
        device = x_start.device
        batch = x_start.shape[0]
        num_nodes = x_start.shape[1]
        perm = torch.arange(num_nodes, device=device).expand(batch, -1)

        for step in reversed(self.diffusion_utils.reverse_steps[1:]):
            t = torch.full((batch,), step, device=device)
            scores = self.reverse_model(
                perm.unsqueeze(-2),
                t,
                x_start,
                priority_start,
            ).squeeze(-3)
            sample_indices = PL.sample_generalized_PL(scores, deterministic=deterministic)
            perm = _sd_utils.permute_int_list(sample_indices, perm)

        result_x = _sd_utils.permute_embd(perm, x_start)
        return result_x, perm

    @torch.no_grad()
    def _p_sample_beam_search_with_priority(
        self,
        x_start: Tensor,
        priority_start: Tensor,
        return_candidates: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Beam-search reverse diffusion conditioned on fixed priority features.

        This mirrors DiffusionUtils.p_sample_beam_search, but calls the priority
        reverse model with both x_start and priority_start.
        """
        if self.diffusion_utils.reverse != "generalized_PL":
            raise NotImplementedError("Priority beam search currently supports generalized_PL only.")

        device = x_start.device
        batch_size = x_start.size(0)
        num_nodes = x_start.size(1)
        pl_beam_size = self.diffusion_utils.PL_beam_size
        t_beam_size = self.diffusion_utils.t_beam_size

        time = torch.full(
            (batch_size,),
            self.diffusion_utils.num_timesteps,
            device=device,
        )
        identity_perm = torch.arange(num_nodes, device=device).expand(batch_size, 1, -1)
        scores = self.reverse_model(
            identity_perm,
            time,
            x_start,
            priority_start,
        ).squeeze(1)

        result_perm, result_log_probs = PL.sample_generalized_PL_beam_search(
            scores,
            pl_beam_size,
        )
        result_perm = result_perm[..., :t_beam_size, :]
        result_log_probs = result_log_probs[..., :t_beam_size]

        for step in reversed(self.diffusion_utils.reverse_steps[1:-1]):
            time = torch.full((batch_size,), step, device=device)
            scores = self.reverse_model(
                result_perm,
                time,
                x_start,
                priority_start,
            )
            candidates_perm, candidates_log_probs = PL.sample_generalized_PL_beam_search(
                scores,
                pl_beam_size,
            )
            candidates_perm = candidates_perm[..., :t_beam_size, :]
            candidates_log_probs = candidates_log_probs[..., :t_beam_size]

            candidates_perm = _sd_utils.permute_int_list(
                candidates_perm,
                result_perm.unsqueeze(-2),
            )
            candidates_log_probs = result_log_probs.unsqueeze(-1) + candidates_log_probs

            candidates_perm = candidates_perm.flatten(start_dim=-3, end_dim=-2)
            candidates_log_probs = candidates_log_probs.flatten(start_dim=-2)

            num_selected = min(t_beam_size, candidates_log_probs.size(-1))
            result_log_probs, topk_idx = torch.topk(
                candidates_log_probs,
                k=num_selected,
                dim=-1,
            )
            topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, num_nodes)
            result_perm = torch.gather(candidates_perm, -2, topk_idx_expanded)

        if return_candidates:
            return result_perm, result_log_probs

        best_perm = result_perm[:, 0, :]
        result_x = _sd_utils.permute_embd(best_perm, x_start)
        return result_x, best_perm

    def forward(
        self,
        target_data: Tensor,
        graph: Optional[Tensor],
        mask: Optional[Tensor] = None,
        is_training: bool = True,
    ):
        if graph is None:
            node_repr = self._encode_raw_data(target_data, mask=mask)
            priority = self._sample_priorities(
                batch_size=node_repr.size(0),
                num_nodes=node_repr.size(1),
                device=node_repr.device,
                dtype=node_repr.dtype,
            )
            was_training = self.reverse_model.training
            self.reverse_model.eval()
            try:
                _, orders = self._p_sample_loop_with_priority(
                    node_repr,
                    priority_start=priority,
                    deterministic=True,
                )
            finally:
                self.reverse_model.train(was_training)
            return {"orders": orders, "priority": priority}

        node_repr, priority_ordered, clean_orders, priority = self._encode_priority_ordered_data(
            target_data,
            graph=graph,
            mask=mask,
        )
        loss = self._training_losses_per_batch_with_priority(
            node_repr,
            priority_start=priority_ordered,
        )
        return {
            "loss": loss,
            "clean_orders": clean_orders,
            "priority": priority,
        }

    def sample(self, target_data: Tensor, num_samples: int = 1, mask: Optional[Tensor] = None):
        node_repr = self._encode_raw_data(target_data, mask=mask)
        batch_size, num_nodes, d_model = node_repr.shape
        priority = self._sample_priorities(
            batch_size=num_samples * batch_size,
            num_nodes=num_nodes,
            device=node_repr.device,
            dtype=node_repr.dtype,
        )
        node_repr = (
            node_repr.unsqueeze(0)
            .expand(num_samples, batch_size, num_nodes, d_model)
            .reshape(num_samples * batch_size, num_nodes, d_model)
        )
        was_training = self.reverse_model.training
        self.reverse_model.eval()
        try:
            _, orders = self._p_sample_loop_with_priority(
                node_repr,
                priority_start=priority,
                deterministic=False,
            )
        finally:
            self.reverse_model.train(was_training)
        orders = orders.reshape(num_samples, batch_size, num_nodes)
        priority = priority.reshape(num_samples, batch_size, num_nodes)
        return orders, priority


class CausalTopoOrderDiffusionWithSkeleton(BakStyleSkeletonMixin, CausalTopoOrderDiffusion):
    """Topo-order diffusion plus a bak-style unordered skeleton head."""

    def __init__(
        self,
        *args,
        skeleton_loss_weight: float = 1.0,
        order_loss_weight: float = 1.0,
        skeleton_decoder_layers: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_skeleton_head(
            d_model=kwargs["d_model"],
            nhead=kwargs["nhead"],
            dim_feedforward=kwargs["dim_feedforward"],
            num_layers_skeleton=skeleton_decoder_layers,
            dropout=kwargs.get("dropout", 0.0),
            skeleton_loss_weight=skeleton_loss_weight,
            order_loss_weight=order_loss_weight,
            device=kwargs.get("device", None),
            dtype=kwargs.get("dtype", None),
        )

    def forward(
        self,
        target_data: Tensor,
        graph: Optional[Tensor],
        mask: Optional[Tensor] = None,
        is_training: bool = True,
    ):
        if graph is None:
            output = super().forward(target_data, graph=graph, mask=mask, is_training=is_training)
            output["skeleton_logits"] = self._skeleton_logits_from_data(target_data, mask=mask)
            return output

        clean_orders = None
        if self.order_loss_weight != 0:
            node_repr, clean_orders = self._encode_ordered_data(
                target_data,
                graph=graph,
                mask=mask,
            )
            order_loss = self._training_losses_per_batch(node_repr)
        else:
            order_loss = torch.zeros(target_data.size(0), device=target_data.device, dtype=target_data.dtype)

        if self.skeleton_loss_weight != 0:
            skeleton_logits = self._skeleton_logits_from_data(target_data, mask=mask)
            skeleton_loss = self._skeleton_loss_per_batch(
                skeleton_logits,
                graph=graph,
                mask=mask,
            )
        else:
            skeleton_logits = None
            skeleton_loss = torch.zeros_like(order_loss)
        return {
            "loss": self.order_loss_weight * order_loss + self.skeleton_loss_weight * skeleton_loss,
            "order_loss": order_loss,
            "skeleton_loss": skeleton_loss,
            "clean_orders": clean_orders,
            "skeleton_logits": skeleton_logits,
        }


class CausalPriorityTopoOrderDiffusionWithSkeleton(
    BakStyleSkeletonMixin,
    CausalPriorityTopoOrderDiffusion,
):
    """Priority-conditioned topo diffusion plus a bak-style skeleton head."""

    def __init__(
        self,
        *args,
        skeleton_loss_weight: float = 1.0,
        order_loss_weight: float = 1.0,
        skeleton_decoder_layers: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_skeleton_head(
            d_model=kwargs["d_model"],
            nhead=kwargs["nhead"],
            dim_feedforward=kwargs["dim_feedforward"],
            num_layers_skeleton=skeleton_decoder_layers,
            dropout=kwargs.get("dropout", 0.0),
            skeleton_loss_weight=skeleton_loss_weight,
            order_loss_weight=order_loss_weight,
            device=kwargs.get("device", None),
            dtype=kwargs.get("dtype", None),
        )

    def forward(
        self,
        target_data: Tensor,
        graph: Optional[Tensor],
        mask: Optional[Tensor] = None,
        is_training: bool = True,
    ):
        if graph is None:
            output = super().forward(target_data, graph=graph, mask=mask, is_training=is_training)
            output["skeleton_logits"] = self._skeleton_logits_from_data(target_data, mask=mask)
            return output

        clean_orders = None
        priority = None
        if self.order_loss_weight != 0:
            node_repr, priority_ordered, clean_orders, priority = self._encode_priority_ordered_data(
                target_data,
                graph=graph,
                mask=mask,
            )
            order_loss = self._training_losses_per_batch_with_priority(
                node_repr,
                priority_start=priority_ordered,
            )
        else:
            order_loss = torch.zeros(target_data.size(0), device=target_data.device, dtype=target_data.dtype)

        if self.skeleton_loss_weight != 0:
            skeleton_logits = self._skeleton_logits_from_data(target_data, mask=mask)
            skeleton_loss = self._skeleton_loss_per_batch(
                skeleton_logits,
                graph=graph,
                mask=mask,
            )
        else:
            skeleton_logits = None
            skeleton_loss = torch.zeros_like(order_loss)
        return {
            "loss": self.order_loss_weight * order_loss + self.skeleton_loss_weight * skeleton_loss,
            "order_loss": order_loss,
            "skeleton_loss": skeleton_loss,
            "clean_orders": clean_orders,
            "priority": priority,
            "skeleton_logits": skeleton_logits,
        }


class CausalTopoOrderDiffusionSingleEncoderWithSkeleton(
    PrecedenceRelationMixin,
    CausalTopoOrderDiffusionWithSkeleton,
):
    """Joint topo+skeleton model that encodes the raw dataset only once."""

    def __init__(
        self,
        *args,
        topo_precedence_loss_weight: float = 0.0,
        topo_precedence_hidden_dim: int = 64,
        topo_precedence_rerank_beta: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_precedence_relation(
            d_model=kwargs["d_model"],
            hidden_dim=topo_precedence_hidden_dim,
            loss_weight=topo_precedence_loss_weight,
            rerank_beta=topo_precedence_rerank_beta,
            device=kwargs.get("device", None),
            dtype=kwargs.get("dtype", None),
        )

    def forward(
        self,
        target_data: Tensor,
        graph: Optional[Tensor],
        mask: Optional[Tensor] = None,
        is_training: bool = True,
    ):
        raw_node_repr = self._encode_raw_data(target_data, mask=mask)

        if graph is None:
            precedence_logits = (
                self.precedence_head(raw_node_repr)
                if self.precedence_head is not None
                else None
            )
            was_training = self.reverse_model.training
            self.reverse_model.eval()
            try:
                if self.precedence_head is not None and self.topo_precedence_rerank_beta > 0:
                    orders, _ = self.sample_precedence_reranked_beam_from_node_repr(
                        raw_node_repr,
                        num_samples=1,
                        mask=mask,
                    )
                else:
                    _, orders = self.diffusion_utils.p_sample_loop(
                        raw_node_repr,
                        self.reverse_model,
                        deterministic=True,
                    )
            finally:
                self.reverse_model.train(was_training)
            return {
                "orders": orders,
                "skeleton_logits": self._skeleton_logits_from_node_repr(
                    raw_node_repr,
                    mask=mask,
                ),
                "precedence_logits": precedence_logits,
            }

        clean_orders = None
        if self.order_loss_weight != 0:
            clean_orders = self._sample_batch_topological_orders(graph, mask=mask)
            ordered_node_repr = self._reorder_node_repr(raw_node_repr, clean_orders)
            order_loss = self._training_losses_per_batch(ordered_node_repr)
        else:
            order_loss = torch.zeros(
                target_data.size(0),
                device=target_data.device,
                dtype=target_data.dtype,
            )

        if self.skeleton_loss_weight != 0:
            skeleton_logits = self._skeleton_logits_from_node_repr(
                raw_node_repr,
                mask=mask,
            )
            skeleton_loss = self._skeleton_loss_per_batch(
                skeleton_logits,
                graph=graph,
                mask=mask,
            )
        else:
            skeleton_logits = None
            skeleton_loss = torch.zeros_like(order_loss)

        precedence_logits, precedence_loss = self._precedence_outputs(
            raw_node_repr,
            graph=graph,
            mask=mask,
        )

        return {
            "loss": (
                self.order_loss_weight * order_loss
                + self.skeleton_loss_weight * skeleton_loss
                + self.topo_precedence_loss_weight * precedence_loss
            ),
            "order_loss": order_loss,
            "skeleton_loss": skeleton_loss,
            "precedence_loss": precedence_loss,
            "clean_orders": clean_orders,
            "skeleton_logits": skeleton_logits,
            "precedence_logits": precedence_logits,
        }


class CausalPriorityTopoOrderDiffusionSingleEncoderWithSkeleton(
    PrecedenceRelationMixin,
    CausalPriorityTopoOrderDiffusionWithSkeleton,
):
    """Priority-conditioned joint model that encodes the raw dataset only once."""

    def __init__(
        self,
        *args,
        topo_precedence_loss_weight: float = 0.0,
        topo_precedence_hidden_dim: int = 64,
        topo_precedence_rerank_beta: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_precedence_relation(
            d_model=kwargs["d_model"],
            hidden_dim=topo_precedence_hidden_dim,
            loss_weight=topo_precedence_loss_weight,
            rerank_beta=topo_precedence_rerank_beta,
            device=kwargs.get("device", None),
            dtype=kwargs.get("dtype", None),
        )

    def forward(
        self,
        target_data: Tensor,
        graph: Optional[Tensor],
        mask: Optional[Tensor] = None,
        is_training: bool = True,
    ):
        raw_node_repr = self._encode_raw_data(target_data, mask=mask)

        if graph is None:
            precedence_logits = (
                self.precedence_head(raw_node_repr)
                if self.precedence_head is not None
                else None
            )
            priority = self._sample_priorities(
                batch_size=raw_node_repr.size(0),
                num_nodes=raw_node_repr.size(1),
                device=raw_node_repr.device,
                dtype=raw_node_repr.dtype,
            )
            was_training = self.reverse_model.training
            self.reverse_model.eval()
            try:
                if self.precedence_head is not None and self.topo_precedence_rerank_beta > 0:
                    orders, priority = self.sample_precedence_reranked_beam_from_node_repr(
                        raw_node_repr,
                        num_samples=1,
                        mask=mask,
                    )
                else:
                    _, orders = self._p_sample_loop_with_priority(
                        raw_node_repr,
                        priority_start=priority,
                        deterministic=True,
                    )
            finally:
                self.reverse_model.train(was_training)
            return {
                "orders": orders,
                "priority": priority,
                "skeleton_logits": self._skeleton_logits_from_node_repr(
                    raw_node_repr,
                    mask=mask,
                ),
                "precedence_logits": precedence_logits,
            }

        clean_orders = None
        priority = None
        if self.order_loss_weight != 0:
            batch_size, num_nodes = raw_node_repr.shape[:2]
            priority = self._sample_priorities(
                batch_size=batch_size,
                num_nodes=num_nodes,
                device=raw_node_repr.device,
                dtype=raw_node_repr.dtype,
            )
            clean_orders = self._sample_batch_priority_topological_orders(
                graph,
                priority=priority,
                mask=mask,
            )
            ordered_node_repr = self._reorder_node_repr(raw_node_repr, clean_orders)
            priority_ordered = self._reorder_priority(priority, clean_orders)
            order_loss = self._training_losses_per_batch_with_priority(
                ordered_node_repr,
                priority_start=priority_ordered,
            )
        else:
            order_loss = torch.zeros(
                target_data.size(0),
                device=target_data.device,
                dtype=target_data.dtype,
            )

        if self.skeleton_loss_weight != 0:
            skeleton_logits = self._skeleton_logits_from_node_repr(
                raw_node_repr,
                mask=mask,
            )
            skeleton_loss = self._skeleton_loss_per_batch(
                skeleton_logits,
                graph=graph,
                mask=mask,
            )
        else:
            skeleton_logits = None
            skeleton_loss = torch.zeros_like(order_loss)

        precedence_logits, precedence_loss = self._precedence_outputs(
            raw_node_repr,
            graph=graph,
            mask=mask,
        )

        return {
            "loss": (
                self.order_loss_weight * order_loss
                + self.skeleton_loss_weight * skeleton_loss
                + self.topo_precedence_loss_weight * precedence_loss
            ),
            "order_loss": order_loss,
            "skeleton_loss": skeleton_loss,
            "precedence_loss": precedence_loss,
            "clean_orders": clean_orders,
            "priority": priority,
            "skeleton_logits": skeleton_logits,
            "precedence_logits": precedence_logits,
        }


class CausalSourceLayerJointSkeletonSingleEncoder(
    SourceLayerMixin,
    BakStyleSkeletonMixin,
    CausalTNPEncoder,
):
    """Joint source-layer and skeleton model with one shared data encoder."""

    def __init__(
        self,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers_encoder,
        num_layers_decoder,
        num_nodes,
        n_perm_samples=None,
        sinkhorn_iter=None,
        use_positional_encoding=False,
        skeleton_loss_weight: float = 1.0,
        order_loss_weight: float = 1.0,
        skeleton_decoder_layers: int = 2,
        source_threshold: float = 0.5,
        source_pos_weight: float = 1.0,
        skeleton_threshold: float = 0.5,
        source_use_global_context: bool = True,
        device=None,
        dtype=None,
        mlp_use_bias: bool = False,
        **kwargs,
    ):
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
            mlp_use_bias=mlp_use_bias,
        )
        self.num_nodes = num_nodes
        self.source_threshold = source_threshold
        self.source_pos_weight = source_pos_weight
        self.skeleton_threshold = skeleton_threshold
        self._init_skeleton_head(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers_skeleton=skeleton_decoder_layers,
            dropout=dropout,
            skeleton_loss_weight=skeleton_loss_weight,
            order_loss_weight=order_loss_weight,
            device=device,
            dtype=dtype,
        )
        self.source_layer_decoder = SourceLayerDecoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers_decoder,
            dropout=dropout,
            use_global_context=source_use_global_context,
            device=device,
            dtype=dtype,
        )

    def _encode_raw_data(self, target_data: Tensor, mask: Optional[Tensor]) -> Tensor:
        if target_data.dim() == 3:
            target_data = target_data.unsqueeze(-1)
        return self.encode(target_data=target_data, mask=mask).squeeze(2)

    def forward(
        self,
        target_data: Tensor,
        graph: Optional[Tensor],
        mask: Optional[Tensor] = None,
        is_training: bool = True,
    ):
        raw_node_repr = self._encode_raw_data(target_data, mask=mask)
        batch_size, num_nodes = raw_node_repr.shape[:2]
        valid_nodes = self._valid_nodes_from_decoder_mask(
            mask,
            batch_size=batch_size,
            num_nodes=num_nodes,
            device=raw_node_repr.device,
        )
        skeleton_logits = self._skeleton_logits_from_node_repr(raw_node_repr, mask=mask)

        if graph is None:
            layer_ids, layer_counts, _ = self._decode_source_layers_from_node_repr(
                raw_node_repr,
                valid_nodes=valid_nodes,
            )
            orders = self._orders_from_layer_ids(layer_ids, valid_nodes)
            adjacency = self._orient_skeleton_by_layer_ids(
                skeleton_logits,
                layer_ids=layer_ids,
                valid_nodes=valid_nodes,
                threshold=self.skeleton_threshold,
            )
            return {
                "orders": orders,
                "layer_ids": layer_ids,
                "layer_counts": layer_counts,
                "skeleton_logits": skeleton_logits,
                "adjacency": adjacency,
            }

        if self.order_loss_weight != 0:
            source_loss, true_layer_ids, source_targets, layer_counts = self._source_layer_loss_per_batch(
                raw_node_repr,
                graph=graph,
                mask=mask,
            )
        else:
            source_loss = torch.zeros(
                target_data.size(0),
                device=target_data.device,
                dtype=target_data.dtype,
            )
            true_layer_ids = None
            source_targets = None
            layer_counts = None

        if self.skeleton_loss_weight != 0:
            skeleton_loss = self._skeleton_loss_per_batch(
                skeleton_logits,
                graph=graph,
                mask=mask,
            )
        else:
            skeleton_loss = torch.zeros_like(source_loss)

        return {
            "loss": self.order_loss_weight * source_loss + self.skeleton_loss_weight * skeleton_loss,
            "order_loss": source_loss,
            "source_layer_loss": source_loss,
            "skeleton_loss": skeleton_loss,
            "true_layer_ids": true_layer_ids,
            "source_targets": source_targets,
            "layer_counts": layer_counts,
            "skeleton_logits": skeleton_logits,
        }

    def calculate_loss(self, output, target):
        if not isinstance(output, dict) or "loss" not in output:
            raise ValueError(
                "CausalSourceLayerJointSkeletonSingleEncoder expects forward() "
                "to return a loss dict."
            )
        return output["loss"]

    def predict_adjacency(
        self,
        target_data: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        output = self.forward(target_data=target_data, graph=None, mask=mask, is_training=False)
        return output

    def sample(self, target_data: Tensor, num_samples: int = 1, mask: Optional[Tensor] = None):
        output = self.predict_adjacency(target_data=target_data, mask=mask)
        orders = output["orders"].unsqueeze(0).expand(num_samples, -1, -1).contiguous()
        info = {
            "layer_ids": output["layer_ids"],
            "layer_counts": output["layer_counts"],
            "adjacency": output["adjacency"],
            "skeleton_logits": output["skeleton_logits"],
        }
        return orders, info


class CausalPriorityNodeTopoOrderDiffusionSingleEncoderWithSkeleton(
    CausalPriorityTopoOrderDiffusionSingleEncoderWithSkeleton,
):
    """Single-encoder joint model with priority fused into topo node embeddings.

    The skeleton head still consumes the raw causal encoder representation in
    the original node order. The topo diffusion branch first fuses each node's
    priority into that node representation, then reorders the fused topo
    representation by the clean priority-Kahn order before applying the
    diffusion loss.
    """

    def __init__(
        self,
        *args,
        topo_priority_node_emb_dim: int = 16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        d_model = kwargs.get("d_model", args[0] if args else None)
        if d_model is None:
            raise ValueError("d_model is required for priority node fusion.")

        factory_kwargs = {}
        if kwargs.get("device", None) is not None:
            factory_kwargs["device"] = kwargs["device"]
        if kwargs.get("dtype", None) is not None:
            factory_kwargs["dtype"] = kwargs["dtype"]

        self.topo_priority_node_emb_dim = topo_priority_node_emb_dim
        self.topo_priority_node_embedder = nn.Sequential(
            nn.Linear(2, topo_priority_node_emb_dim, **factory_kwargs),
            nn.GELU(),
            nn.Linear(topo_priority_node_emb_dim, topo_priority_node_emb_dim, **factory_kwargs),
        )
        self.topo_priority_node_fuse = nn.Sequential(
            nn.Linear(d_model + topo_priority_node_emb_dim, d_model, **factory_kwargs),
            nn.GELU(),
            nn.Linear(d_model, d_model, **factory_kwargs),
        )

    @staticmethod
    def _priority_node_features(priority: Tensor) -> Tensor:
        priority = priority.float()
        rank = priority.argsort(dim=-1).argsort(dim=-1).to(dtype=priority.dtype)
        denom = max(priority.size(-1) - 1, 1)
        rank = rank / denom
        return torch.stack([priority, rank], dim=-1)

    def _fuse_priority_into_topo_node_repr(
        self,
        raw_node_repr: Tensor,
        priority: Tensor,
    ) -> Tensor:
        priority_features = self._priority_node_features(priority).to(
            device=raw_node_repr.device,
            dtype=raw_node_repr.dtype,
        )
        priority_emb = self.topo_priority_node_embedder(priority_features)
        return self.topo_priority_node_fuse(
            torch.cat([raw_node_repr, priority_emb], dim=-1)
        )

    def forward(
        self,
        target_data: Tensor,
        graph: Optional[Tensor],
        mask: Optional[Tensor] = None,
        is_training: bool = True,
    ):
        raw_node_repr = self._encode_raw_data(target_data, mask=mask)

        if graph is None:
            priority = self._sample_priorities(
                batch_size=raw_node_repr.size(0),
                num_nodes=raw_node_repr.size(1),
                device=raw_node_repr.device,
                dtype=raw_node_repr.dtype,
            )
            topo_node_repr = self._fuse_priority_into_topo_node_repr(
                raw_node_repr,
                priority,
            )
            was_training = self.reverse_model.training
            self.reverse_model.eval()
            try:
                _, orders = self._p_sample_loop_with_priority(
                    topo_node_repr,
                    priority_start=priority,
                    deterministic=True,
                )
            finally:
                self.reverse_model.train(was_training)
            return {
                "orders": orders,
                "priority": priority,
                "skeleton_logits": self._skeleton_logits_from_node_repr(
                    raw_node_repr,
                    mask=mask,
                ),
            }

        clean_orders = None
        priority = None
        if self.order_loss_weight != 0:
            batch_size, num_nodes = raw_node_repr.shape[:2]
            priority = self._sample_priorities(
                batch_size=batch_size,
                num_nodes=num_nodes,
                device=raw_node_repr.device,
                dtype=raw_node_repr.dtype,
            )
            clean_orders = self._sample_batch_priority_topological_orders(
                graph,
                priority=priority,
                mask=mask,
            )
            topo_node_repr = self._fuse_priority_into_topo_node_repr(
                raw_node_repr,
                priority,
            )
            ordered_topo_node_repr = self._reorder_node_repr(
                topo_node_repr,
                clean_orders,
            )
            priority_ordered = self._reorder_priority(priority, clean_orders)
            order_loss = self._training_losses_per_batch_with_priority(
                ordered_topo_node_repr,
                priority_start=priority_ordered,
            )
        else:
            order_loss = torch.zeros(
                target_data.size(0),
                device=target_data.device,
                dtype=target_data.dtype,
            )

        if self.skeleton_loss_weight != 0:
            skeleton_logits = self._skeleton_logits_from_node_repr(
                raw_node_repr,
                mask=mask,
            )
            skeleton_loss = self._skeleton_loss_per_batch(
                skeleton_logits,
                graph=graph,
                mask=mask,
            )
        else:
            skeleton_logits = None
            skeleton_loss = torch.zeros_like(order_loss)

        return {
            "loss": self.order_loss_weight * order_loss + self.skeleton_loss_weight * skeleton_loss,
            "order_loss": order_loss,
            "skeleton_loss": skeleton_loss,
            "clean_orders": clean_orders,
            "priority": priority,
            "skeleton_logits": skeleton_logits,
        }

    def sample(self, target_data: Tensor, num_samples: int = 1, mask: Optional[Tensor] = None):
        raw_node_repr = self._encode_raw_data(target_data, mask=mask)
        batch_size, num_nodes, d_model = raw_node_repr.shape
        priority = self._sample_priorities(
            batch_size=num_samples * batch_size,
            num_nodes=num_nodes,
            device=raw_node_repr.device,
            dtype=raw_node_repr.dtype,
        )
        topo_node_repr = (
            raw_node_repr.unsqueeze(0)
            .expand(num_samples, batch_size, num_nodes, d_model)
            .reshape(num_samples * batch_size, num_nodes, d_model)
        )
        topo_node_repr = self._fuse_priority_into_topo_node_repr(
            topo_node_repr,
            priority,
        )
        was_training = self.reverse_model.training
        self.reverse_model.eval()
        try:
            _, orders = self._p_sample_loop_with_priority(
                topo_node_repr,
                priority_start=priority,
                deterministic=False,
            )
        finally:
            self.reverse_model.train(was_training)
        orders = orders.reshape(num_samples, batch_size, num_nodes)
        priority = priority.reshape(num_samples, batch_size, num_nodes)
        return orders, priority
