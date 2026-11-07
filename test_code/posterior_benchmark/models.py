"""Raw-X encoders and exactly enumerated three-node graph distributions.

No thresholding, consensus order, balanced BCE, teacher posterior, or scatter
features enter these models. Diffusion training can reuse the original sampled
riffle/denoising objective; evaluation always enumerates the full graph law.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ml2_meta_causal_discovery.models.causaltransformercomponents import CausalTNPEncoder
from ml2_meta_causal_discovery.models.topo_order_diffusion import (
    BakStyleSkeletonHead, BakStyleSkeletonMixin, CausalEmbeddingReverseDiffusion,
    CausalTopoOrderDiffusion, CausalPriorityTopoOrderDiffusion,
    PriorityCausalEmbeddingReverseDiffusion, DiffusionUtils, PL,
)

METHODS = ("categorical", "diffusion", "pl")
PAIRS = ((0, 1), (0, 2), (1, 2))
PERMUTATIONS = tuple(itertools.permutations(range(3)))


@dataclass(frozen=True)
class ModelConfig:
    dim_model: int = 256
    dim_feedforward: int = 512
    nhead: int = 8
    encoder_layers: int = 4
    skeleton_layers: int = 2
    denoiser_layers: int = 7
    dropout: float = 0.1
    reverse_steps: tuple[int, ...] = (0, 4, 7, 11)
    # Zero: exact six-state model without priority. Positive: a fixed uniform-u
    # Monte Carlo bank, shared by PL/GPL, integrated BEFORE graph NLL / KL.
    priority_samples: int = 0
    priority_seed: int = 20261107

    def validate(self):
        if self.dim_model <= 0 or self.nhead <= 0 or self.dim_model % self.nhead:
            raise ValueError("dim_model must be positive and divisible by nhead")
        if self.encoder_layers < 2 or self.encoder_layers % 2:
            raise ValueError("The existing causal encoder requires a positive even layer count")
        if min(self.dim_feedforward, self.skeleton_layers, self.denoiser_layers) < 1:
            raise ValueError("Feedforward width and decoder depths must be positive")
        if not 0 <= self.dropout < 1 or self.priority_samples < 0 or self.priority_seed < 0:
            raise ValueError("Invalid dropout or priority sample count")
        if len(self.reverse_steps) < 2 or self.reverse_steps[0] != 0:
            raise ValueError("reverse_steps must start at zero and contain at least one transition")
        if any(a >= b for a, b in zip(self.reverse_steps, self.reverse_steps[1:])):
            raise ValueError("reverse_steps must be strictly increasing")

    def document(self):
        result = asdict(self)
        result["reverse_steps"] = list(self.reverse_steps)
        return result


def graph_tables(graphs):
    graphs = np.asarray(graphs)
    if graphs.shape != (25, 3, 3):
        raise ValueError("Expected the catalogue's 25 labelled three-node DAGs")
    skeletons = (graphs | graphs.transpose(0, 2, 1)).astype(bool)
    bits = np.stack([skeletons[:, i, j] for i, j in PAIRS], axis=-1)
    compatible = np.zeros((25, 6), dtype=bool)
    for k, order in enumerate(PERMUTATIONS):
        positions = np.argsort(order)
        allowed = positions[:, None] < positions[None, :]
        compatible[:, k] = ~np.any(graphs.astype(bool) & ~allowed, axis=(1, 2))
    if not compatible.any(axis=-1).all():
        raise ValueError("Catalogue contains a cyclic graph")
    return bits, compatible


def graph_log_probabilities(edge_logits, order_log_probs, skeleton_bits, compatible):
    """q(G|X)=q(S_G|X) sum_{pi compatible with G} q(pi|X).

    Every (skeleton, permutation) maps to exactly one of the 25 DAGs. Do NOT
    divide by the number of compatible orders or multiply by another 1/25.
    """
    logits = edge_logits.double().unsqueeze(-2)
    bits = skeleton_bits.to(dtype=torch.float64)
    skeleton_log_probs = (bits * F.logsigmoid(logits)
                          + (1 - bits) * F.logsigmoid(-logits)).sum(-1)
    masked = order_log_probs.double().unsqueeze(-2).masked_fill(~compatible, -torch.inf)
    return skeleton_log_probs + torch.logsumexp(masked, dim=-1)


def propagate_log_distribution(log_distribution, log_transition):
    """Rows are current noisy states; columns are preceding/cleaner states."""
    return torch.logsumexp(log_distribution.unsqueeze(-1) + log_transition, dim=-2)


class PosteriorModel(nn.Module):
    def __init__(self, method, config: ModelConfig, graphs):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown posterior method: {method}")
        config.validate()
        self.method, self.config = method, config
        # Constructed FIRST for all methods: identical seed gives identical
        # initial encoder weights. No new n/scatter/mean/variance input.
        self.encoder = CausalTNPEncoder(
            d_model=config.dim_model, dim_feedforward=config.dim_feedforward,
            nhead=config.nhead, num_layers=config.encoder_layers,
            use_positional_encoding=False, num_nodes=3, device=None, dtype=None,
            emb_depth=1, dropout=config.dropout, mlp_use_bias=False,
        )
        bits, compatible = graph_tables(graphs)
        self.register_buffer("graphs", torch.as_tensor(graphs, dtype=torch.int8))
        self.register_buffer("skeleton_bits", torch.tensor(bits, dtype=torch.bool))
        self.register_buffer("compatible", torch.tensor(compatible))
        self.register_buffer("permutations", torch.tensor(PERMUTATIONS, dtype=torch.long))
        generator = torch.Generator().manual_seed(config.priority_seed)
        bank = torch.rand((config.priority_samples, 3), generator=generator)
        self.register_buffer("priority_bank", bank)
        if method == "categorical":
            self.classifier = nn.Sequential(
                nn.Linear(3 * config.dim_model, config.dim_feedforward), nn.GELU(),
                nn.Dropout(config.dropout), nn.Linear(config.dim_feedforward, 25),
            )
            return
        # Construct skeleton before method-specific order head: same weights
        # at initialization in PL and GPL for a common training seed.
        self.skeleton = BakStyleSkeletonHead(
            config.dim_model, config.nhead, config.dim_feedforward,
            config.skeleton_layers, config.dropout,
        )
        if method == "pl":
            extra = 2 if config.priority_samples else 0
            self.order_head = nn.Sequential(
                nn.Linear(config.dim_model + extra, config.dim_feedforward), nn.GELU(),
                nn.Linear(config.dim_feedforward, 1),
            )
        else:
            denoiser = (PriorityCausalEmbeddingReverseDiffusion if config.priority_samples
                        else CausalEmbeddingReverseDiffusion)
            self.denoiser = denoiser(
                config.dim_model, config.nhead, config.dim_feedforward,
                config.denoiser_layers, config.dropout,
            )
            self.diffusion_utils = DiffusionUtils(
                num_timesteps=config.reverse_steps[-1], sample_N=1, transition="riffle",
                latent=False, reinforce_N=10, reinforce_ema_rate=.995,
                entropy_reg_rate=.05, reverse="generalized_PL",
                reverse_steps=list(config.reverse_steps), loss="log_likelihood",
                beam_size={"PL": 20, "time": 20}, perm_fix_first=False,
            )
            self.skeleton_loss_type = "bce"

    @property
    def reverse_model(self):
        """Adapter for calling the ORIGINAL denoising-loss methods unchanged."""
        return self.denoiser

    _decoder_mask = staticmethod(BakStyleSkeletonMixin._decoder_mask)

    def encode(self, data):
        if data.ndim != 3 or data.shape[-1] != 3 or data.shape[1] < 2:
            raise ValueError("Input must be raw X with shape [batch,n,3], n>=2")
        # Subsampling occurs before this function. Preserve every selected row
        # in A/B/C order; the only conversion here is the model dtype.
        data = data.to(dtype=next(self.encoder.parameters()).dtype)
        return self.encoder.encode(data.unsqueeze(-1), mask=None).squeeze(2)

    def training_loss(self, data, classes, diffusion_loss="denoising",
                      order_loss_weight=.2, skeleton_loss_weight=1.):
        if self.method != "diffusion" or diffusion_loss == "graph_nll":
            loss = graph_nll(self(data), classes)
            return {"loss": loss, "graph_nll": loss}
        if diffusion_loss != "denoising":
            raise ValueError(f"Unknown diffusion objective: {diffusion_loss}")
        representation = self.encode(data)
        graph = self.graphs[classes.long()]
        skeleton_logits = self.skeleton(representation)
        # Reuse the exact original ordinary BCE reduction, not balanced BCE.
        skeleton_loss = BakStyleSkeletonMixin._skeleton_loss_per_batch(
            self, skeleton_logits, graph, mask=None).mean()
        if self.config.priority_samples:
            # Original priority training samples fresh continuous u; the fixed
            # integration bank is used only to approximate the evaluation marginal.
            priority = torch.rand((len(data), 3), device=data.device, dtype=torch.float32)
            order = CausalPriorityTopoOrderDiffusion._batched_priority_kahn_topological_orders(
                graph, priority, torch.ones_like(priority, dtype=torch.bool))
            ordered = CausalTopoOrderDiffusion._reorder_node_repr(representation, order)
            priority_ordered = CausalPriorityTopoOrderDiffusion._reorder_priority(priority, order)
            order_loss = CausalPriorityTopoOrderDiffusion._training_losses_per_batch_with_priority(
                self, ordered, priority_ordered).mean()
        else:
            order = CausalTopoOrderDiffusion._sample_batch_topological_orders(self, graph)
            ordered = CausalTopoOrderDiffusion._reorder_node_repr(representation, order)
            # This calls q_sample_seq, selects 0/4/7/11, and computes the original
            # multi-step GPL conditional denoising likelihood. No new DP objective.
            order_loss = CausalTopoOrderDiffusion._training_losses_per_batch(self, ordered).mean()
        return {"loss": order_loss_weight * order_loss + skeleton_loss_weight * skeleton_loss,
                "order_loss": order_loss, "skeleton_loss": skeleton_loss}

    def transition_log_probs(self, node_repr, time, priority=None):
        batch = len(node_repr)
        current = self.permutations.unsqueeze(0).expand(batch, -1, -1)
        timestep = torch.full((batch,), time, device=node_repr.device, dtype=torch.long)
        if priority is None:
            scores = self.denoiser(current, timestep, node_repr)
        else:
            scores = self.denoiser(current, timestep, node_repr, priority)
        # [B,current,next,position,candidate], broadcasting all 36 transitions.
        # Reuse the original relative-permutation convention and GPL likelihood.
        return PL.log_prob_perms_generalized_PL(
            scores.float().unsqueeze(2),
            self.permutations.view(1, 1, 6, 3),
            self.permutations.view(1, 6, 1, 3),
        ).double()

    def _conditional_order_log_probs(self, node_repr, priority=None):
        if self.method == "pl":
            features = node_repr
            if priority is not None:
                extra = PriorityCausalEmbeddingReverseDiffusion._priority_features(priority)
                features = torch.cat((features, extra.to(features.dtype)), dim=-1)
            scores = self.order_head(features).squeeze(-1)
            return PL.log_prob(scores.float().unsqueeze(1),
                               self.permutations.unsqueeze(0)).double()
        log_distribution = node_repr.new_full((len(node_repr), 6), -math.log(6), dtype=torch.float64)
        for time in reversed(self.config.reverse_steps[1:]):
            transitions = self.transition_log_probs(node_repr, time, priority)
            log_distribution = propagate_log_distribution(log_distribution, transitions)
        return log_distribution

    def order_log_probs(self, node_repr):
        if not self.config.priority_samples:
            return self._conditional_order_log_probs(node_repr)
        # A fixed bank makes train/validation/test use the SAME normalized mixture.
        # Exact state/path summation per u; finite-MC approximation over continuous u.
        terms = [self._conditional_order_log_probs(
            node_repr, priority.unsqueeze(0).expand(len(node_repr), -1))
            for priority in self.priority_bank]
        return torch.logsumexp(torch.stack(terms), dim=0) - math.log(len(terms))

    def forward(self, data):
        node_repr = self.encode(data)
        if self.method == "categorical":
            return F.log_softmax(self.classifier(node_repr.flatten(1)).double(), dim=-1)
        skeleton_logits = self.skeleton(node_repr)
        edge_logits = torch.stack([skeleton_logits[:, i, j] for i, j in PAIRS], dim=-1)
        log_probs = graph_log_probabilities(edge_logits, self.order_log_probs(node_repr),
                                            self.skeleton_bits, self.compatible)
        # Original PL/GPL kernels are float32; remove only accumulated roundoff
        # (~1e-7) after the analytically normalized marginalization.
        return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)


def graph_nll(log_probs, classes):
    """Training uses sampled graph labels ONLY; no exact-posterior teacher."""
    return -log_probs.gather(-1, classes.long().unsqueeze(-1)).squeeze(-1).mean()
