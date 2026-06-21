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
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ml2_meta_causal_discovery.models.causaltransformercomponents import (
    CausalAdjacencyMatrix,
    CausalTNPEncoder,
    CausalTransformerDecoderLayer,
)
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
        return self.skeleton_head(
            raw_node_repr,
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
        device=None,
        dtype=None,
        mlp_use_bias: bool = False,
        **kwargs,
    ):
        if topo_reverse != "generalized_PL":
            raise ValueError("CausalPriorityTopoOrderDiffusion currently supports generalized_PL only.")
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
        return torch.rand((batch_size, num_nodes), device=device, dtype=torch.float32)

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
