import copy
import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.parameter import Parameter

from ml2_meta_causal_discovery.models.causaltransformercomponents import (
    CausalAdjacencyMatrix, CausalTNPEncoder, CausalTransformerDecoderLayer,
    build_mlp)
from ml2_meta_causal_discovery.utils.metrics import cyclicity
from ml2_meta_causal_discovery.utils.permutations import (sample_permutation,
                                                          sinkhorn)
from ml2_meta_causal_discovery.utils.topological_orders import (
    orders_to_bak_permutation, sample_topological_orders)


class AviciDecoder(CausalTNPEncoder):

    """
    Differences:
    - Max pool for summary representation in encoder
    - No decoder
    - Linear layer after which attention operation gives adjacency matrix
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
        use_positional_encoding,
        num_nodes,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super(AviciDecoder, self).__init__(
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
            avici_summary=True, # This is the only difference in encoding
        )
        # Decoder is a linear layer.
        # The linear layer with heads is implemented in CausalAdjacencyMatrix
        self.decoder = nn.Identity()

        self.predictor = CausalAdjacencyMatrix(
            nhead=1, # There is only one head for the final prediction
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.regulariser_weight = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.regulariser_lr = 1e-4 # hard coded from avici paper
        self.cyclicity_value_avg = None

    def decode(self, representation):
        # shape [batch_size, num_nodes, d_model]
        decoder_rep = self.decoder(representation)
        return decoder_rep

    def update_regulariser_weight(self, acyclic_loss):
        """
        Should update every 250 steps.
        """
        self.regulariser_weight.data = self.regulariser_weight.data + self.regulariser_lr * acyclic_loss

    def calculate_loss(self, logits, target, update_regulariser=False):
        """
        Args:
        -----
            logits: torch.Tensor, shape [batch_size, num_samples, num_nodes, num_nodes]
            target: torch.Tensor, shape [batch_size, num_nodes, num_nodes]

        Returns:
        --------
            loss: torch.Tensor, shape [batch_size]
            logits: torch.Tensor, shape [batch_size, num_nodes ** 2]
        """
        probs = torch.sigmoid(logits)
        # set diagonal to 0
        probs = probs * (1 - torch.eye(probs.size(-1), device=probs.device))
        current_value = cyclicity(probs).mean().item()  # Get scalar value

        logits = logits.contiguous().view(logits.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        # Classification loss
        loss_func = torch.nn.BCEWithLogitsLoss(reduction="none")
        loss = loss_func(logits, target)
        loss = loss.mean(dim=1)

        # Update EMA of cyclicity_value
        alpha = 0.1  # Smoothing factor between 0 and 1
        if self.cyclicity_value_avg is None:
            self.cyclicity_value_avg = current_value
        else:
            self.cyclicity_value_avg = (
                alpha * current_value + (1 - alpha) * self.cyclicity_value_avg
            )

        acyclic_loss = self.regulariser_weight * current_value

        # Update dual weight with EMA
        if update_regulariser:
            self.update_regulariser_weight(self.cyclicity_value_avg)
        return loss + acyclic_loss

    def forward(self, target_data, graph, is_training=True, mask=None):
        # target_data: [batch_size, num_samples, num_nodes]
        if target_data.dim() == 3:
            target_data = target_data.unsqueeze(-1)
        # Extract representation
        # shape [batch_size, num_nodes, 1, d_model]
        representation = self.encode(target_data=target_data, mask=mask)
        # Decode the representation
        representation = representation.squeeze(2)
        out = self.decode(representation=representation)
        # Final predictor for adjacency matrix
        if mask is not None:
            mask = mask[:, 0, :]
        adj_matrix = self.predictor(out, padding_mask=mask)
        # graph is shape [batch_size, num_nodes, num_nodes]
        # adj_matrix is shape [batch_size, num_nodes, num_nodes]
        return adj_matrix

    def sample(self, target_data: torch.Tensor, num_samples: int, mask=None):
        """
        Sample. num_samples here is samples of the graph.

        Returns:
        --------
            samples: torch.Tensor, shape [num_samples, batch_size, num_nodes, num_nodes]
        """
        adj_matrix = self.forward(target_data, graph=None, is_training=False)
        # Sample from the distribution using a Bernoulli distribution
        existence_dist = torch.distributions.Bernoulli(
            probs=torch.nn.Sigmoid()(adj_matrix)
        )
        # Set diag to 0
        existence_dist.probs = existence_dist.probs * (1 - torch.eye(adj_matrix.size(-1), device=adj_matrix.device))
        samples = existence_dist.sample(
            sample_shape=(num_samples,)
        )
        return samples


class CsivaDecoder(CausalTNPEncoder):

    """"
    Differences:
    - Autoregressive decoder
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
        use_positional_encoding,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super(CsivaDecoder, self).__init__(
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
        )
        self.output_embedder = build_mlp(
            dim_in=1,
            dim_hid=d_model,
            dim_out=d_model,
            depth=emb_depth,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,
                batch_first=True,
                device=device,
                dtype=dtype,
            ),
            num_layers=num_layers_decoder,
        )
        self.predictor = build_mlp(
            dim_in=d_model,
            dim_hid=d_model,
            dim_out=1,
            depth=emb_depth,
        )

    def decode(self, representation, targets, is_training=True):
        # shape [batch_size, num_nodes, d_model]
        if is_training:
            # we will auto-regressively predict the adjacency matrix
            # targets will be the target_graph with -1 as the first one
            # shape [batch_size, num_nodes ** 2, 1]
            target_graph = targets.view(targets.size(0), -1)[:, :, None]
            minus_one_trgt = torch.ones_like(target_graph[:, 0:1, :]) * -1
            full_target = torch.cat([minus_one_trgt, target_graph], dim=1)[:, :-1, :]
            full_target_emb = self.output_embedder(full_target)
            tgt_mask = torch.zeros((full_target_emb.size(1), full_target_emb.size(1)), device=full_target_emb.device).fill_(float('-inf'))
            tgt_mask = tgt_mask.triu_(1)
            decoder_rep = self.decoder(
                tgt=full_target_emb,
                memory=representation,
                tgt_mask=tgt_mask,
            )
            tgt_input = full_target
        else:
            num_nodes = representation.size(1)
            adj_size = num_nodes ** 2
            # shape [batch_size, 1, 1]
            tgt_input = torch.ones_like(representation[:, 0:1, 0:1], device=representation.device) * -1
            while True:
                # shape [batch_size, loop_iteration, d_model]
                tgt_emb = self.output_embedder(tgt_input)
                # Same tgt_mask in validation as for training
                tgt_mask = torch.zeros((tgt_emb.size(1), tgt_emb.size(1))).fill_(float('-inf')).to(representation.device)
                tgt_mask = tgt_mask.triu_(1)
                # size of decoder rep will be [batch, tgt_input.size(1), d_model]
                decoder_rep = self.decoder(
                    tgt=tgt_emb,
                    memory=representation,
                    tgt_mask=tgt_mask,
                )
                # sample bernoulli distribution
                logit = self.predictor(decoder_rep[:, -1:, :])
                # shape [batch_size, 1, 1]
                bernoulli = torch.bernoulli(torch.sigmoid(logit))
                tgt_input = torch.cat([tgt_input, bernoulli], dim=1)
                if tgt_input.size(1) - 1 == adj_size:
                    break
        return decoder_rep, tgt_input

    def calculate_loss(self, logits, target):
        """
        Args:
        -----
            logits: torch.Tensor, shape [batch_size, num_nodes, num_nodes]
            target: torch.Tensor, shape [batch_size, num_nodes, num_nodes]

        Returns:
        --------
            loss: torch.Tensor, shape [batch_size]
            logits: torch.Tensor, shape [batch_size, num_nodes ** 2]
        """
       #  shape [batch_size, num_nodes ** 2]
        logits = logits.contiguous().view(logits.size(0), -1)
        target_graph = target.view(target.size(0), -1)
        # Classification loss
        loss_func = torch.nn.BCEWithLogitsLoss(reduction="none")
        loss = loss_func(logits, target_graph)
        loss = loss.mean(dim=1)
        return loss

    def forward(self, target_data, graph, is_training=True, mask=None):
        """
        Args:
        -----
            target_data: torch.Tensor, shape [batch_size, num_samples, num_nodes]
            graph: torch.Tensor, shape [batch_size, num_nodes, num_nodes]
                This is needed for teacher forcing
            is_training: bool.
                during training, we will use the ground truth adjacency matrix
                but during inference, we will use the predicted adjacency matrix

        Returns:
        --------
            all_logits: torch.Tensor, shape [num_samples, batch_size, num_nodes, num_nodes]
                Logits of the adjacency matrix of the DAG.
        """
        num_nodes = target_data.size(-1)
        # target_data: [batch_size, num_samples, num_nodes]
        if target_data.dim() == 3:
            target_data = target_data.unsqueeze(-1)
        # Extract representation
        # shape [batch_size, num_nodes, 1, d_model]
        representation = self.encode(target_data=target_data, mask=mask)
        # Decode the representation
        # shape [batch_size, num_nodes, d_model]
        representation = representation.squeeze(2)
        # shape [batch_size, num_nodes ** 2, d_model]
        out, predict_graph = self.decode(
            representation=representation,
            targets=graph.clone() if is_training else graph,
            is_training=is_training
        )
        # shape [batch_size, num_nodes ** 2]
        logit = self.predictor(out).squeeze(-1)
        logit = logit.reshape(logit.size(0), num_nodes, num_nodes)

        if is_training:
            return logit
        else:
            predict_graph = predict_graph.squeeze(-1)[:, 1:]
            predict_graph = predict_graph.view(predict_graph.size(0), num_nodes, num_nodes)
            return logit, predict_graph

    def sample(self, target_data: torch.Tensor, num_samples: int, mask=None):
        """
        Sample. num_samples here is samples of the graph.

        Returns:
        --------
            samples: torch.Tensor, shape [num_samples, batch_size, num_nodes, num_nodes]
        """
        all_samples = torch.zeros(
            (num_samples, target_data.size(0), target_data.size(-1), target_data.size(-1)),
            device=target_data.device,
        )
        for i in range(num_samples):
            _, sample = self.forward(target_data, graph=None, is_training=False, mask=mask)
            all_samples[i] = sample
        return all_samples


class CausalProbabilisticDecoder(CausalTNPEncoder):

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
        Q_before_L=True, # Whether to infer Q (perm) before L (bernoulli)
        device=None,
        dtype=None,
        mlp_use_bias: bool = False,
        **kwargs,
    ):
        super(CausalProbabilisticDecoder, self).__init__(
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
        self.n_perm_samples = n_perm_samples
        self.sinkhorn_iter = sinkhorn_iter
        self.Q_before_L = Q_before_L
        self.output_embedder = build_mlp(
            dim_in=1,
            dim_hid=d_model,
            dim_out=d_model,
            depth=emb_depth,
            use_bias=mlp_use_bias,
        )
        # Decoder for the adjacency matrix
        # A = QLQ^Q where L is a lower triangular matrix
        # Q is the permutation matrix
        print(f"Using {num_layers_decoder // 2} decoder layers.")
        self.decoder_L = nn.TransformerDecoder(
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
            num_layers=num_layers_decoder // 2,
        )
        self.decoder_Q = nn.TransformerDecoder(
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
            num_layers=num_layers_decoder // 2,
        )
        self.Q_param = CausalAdjacencyMatrix(
            nhead=nhead,
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.L_param = CausalAdjacencyMatrix(
            nhead=nhead,
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.p_param = build_mlp(
            dim_in=d_model,
            dim_hid=d_model,
            dim_out=1,
            depth=emb_depth,
            use_bias=mlp_use_bias,
        )

    def decode(self, representation, is_training=True, mask: Optional[torch.Tensor] = None):
        if not self.Q_before_L:
            # shape [batch_size, nu`m_nodes, d_model]
            L_rep = self.decoder_L(representation, memory=None, tgt_key_padding_mask=mask)
            # We will pass L_param into permutation
            Q_rep = self.decoder_Q(L_rep, memory=None, tgt_key_padding_mask=mask)
            # shape [batch_size, num_nodes, num_nodes]
            L_param = self.L_param(L_rep, padding_mask=mask)
            # Q_param = self.Q_param(Q_rep)
        else:
            # shape [batch_size, num_nodes, d_model]
            Q_rep = self.decoder_Q(representation, memory=None, tgt_key_padding_mask=mask)
            # We will pass Q_param into L
            L_rep = self.decoder_L(Q_rep, memory=None, tgt_key_padding_mask=mask)
            # shape [batch_size, num_nodes, num_nodes]
            L_param = self.L_param(L_rep, padding_mask=mask)
            # Q_param = self.Q_param(Q_rep)
        # Symmetrize L_param for permutation equivariance
        L_param = (L_param + L_param.transpose(1, 2)) / 2
        return L_param, Q_rep

    def calculate_loss(self, probs, target):
        """
        Args:
        -----
            probs: torch.Tensor, shape [num_samples, batch_size, num_nodes, num_nodes]
            target: torch.Tensor, shape [batch_size, num_nodes, num_nodes]

        Returns:
        --------
            loss: torch.Tensor, shape [batch_size]
        """
        # Reshape the last axis
        probs = probs.contiguous().view(probs.size(0), probs.size(1), -1)
        target_graph = target.reshape(target.size(0), -1)
        # Calculate the loss
        existence_dist = torch.distributions.Bernoulli(
            probs=probs
        )
        log_prob = existence_dist.log_prob(target_graph[None])
        # # Mean across pemutation samples
        log_prob_sum = torch.logsumexp(log_prob, dim=0) - math.log(log_prob.size(0))
        # # shape [batch, num_nodes**2]
        loss_per_edge = - log_prob_sum
        loss = loss_per_edge.mean(dim=1)
        return loss

    def forward(self, target_data, graph, mask: Optional[torch.Tensor] = None, is_training=True):
        """
        Args:
        -----
            target_data: torch.Tensor, shape [batch_size, num_samples, num_nodes]
            is_training: bool.
                during training, we will use the ground truth adjacency matrix
                but during inference, we will use the predicted adjacency matrix
                Only needed for autoregressive model.

        Returns:
        --------
            probs: torch.Tensor, shape [num_samples, batch_size, num_nodes, num_nodes]
                probs of the adjacency matrix of the DAG.
        """
        if self.num_nodes != target_data.size(-1):
            raise ValueError("Number of nodes in the input data should be equal to num_nodes.")
        # target_data: [batch_size, num_samples, num_nodes]
        if target_data.dim() == 3:
            target_data = target_data.unsqueeze(-1)
        # Extract representation
        # shape [batch_size, num_nodes, 1, d_model]
        representation = self.encode(target_data=target_data, mask=mask)
        # Decode the representation
        # shape [batch_size, num_nodes, d_model]
        representation = representation.squeeze(2)
        # L: shape [batch_size, num_nodes, num_nodes]
        # Q: shape [batch_size, num_nodes, d_model]
        if mask is not None:
            decoder_mask = mask[:, 0, :]
        else:
            decoder_mask = None
        L_param, Q_rep = self.decode(representation=representation, mask=decoder_mask)
        # shape [batch_size, num_nodes]
        # we want padded part to have inf
        p_param = self.p_param(Q_rep).squeeze(-1)
        ovector = torch.arange(
            1, self.num_nodes + 1,
            device=p_param.device,
            dtype=p_param.dtype
        )
        Q_param = torch.einsum(
            "bn,m->bnm",
            p_param,
            ovector[: representation.size(1)],
        )
        # Sample permutations
        # shape = [batch_size, n_samples, num_nodes, num_nodes]
        Q_param = torch.functional.F.logsigmoid(Q_param)
        if decoder_mask is not None:
            Q_mask = decoder_mask.unsqueeze(1) + decoder_mask.unsqueeze(2)
            # Set diagonal to 0
            Q_mask = Q_mask * (1 - torch.eye(Q_mask.size(-1), device=Q_mask.device))
            Q_param = Q_param + Q_mask
        perm, _ = sample_permutation(
            log_alpha=Q_param,
            temp=1.0,
            noise_factor=1.0,
            n_samples=self.n_perm_samples,
            hard=True,
            n_iters=self.sinkhorn_iter,
            squeeze=False,
            device=Q_param.device,
        )
        perm = perm.transpose(1, 0)
        perm_inv = torch.transpose(perm, 3, 2)
        # # All matrices
        # extract mask for variable node size
        # tril doesnt work on some dtypes
        mask = torch.tril(
            torch.ones(
                (self.num_nodes, self.num_nodes),
                device=perm.device,
                dtype=torch.float32,
            ),
            diagonal=-1
        ).to(perm.dtype)
        my_mask = mask[: representation.size(1), : representation.size(1)]
        all_masks = torch.einsum(
            "bnij,jk,bnkl->bnil",
            perm,
            my_mask,
            perm_inv,
        )
        # Find probs
        probs = torch.sigmoid(L_param)
        # shape [num_samples, batch_size, num_nodes, num_nodes]
        # Elementwise multiplication
        all_probs = torch.mul(probs[None], all_masks)
        return all_probs

    def sample(self, target_data: torch.Tensor, num_samples: int, mask: Optional[torch.Tensor] = None):
        """
        Sample DAGs, one for each permutation.

        Returns:
        --------
            samples: torch.Tensor, shape [num_samples, batch_size, num_nodes, num_nodes]
        """
        # Override number of samples
        self.n_perm_samples = num_samples
        # probs: [num_samples, batch_size, num_nodes, num_nodes]
        probs = self.forward(target_data, graph=None, is_training=False, mask=mask)
        # Sample from the distribution
        existence_dist = torch.distributions.Bernoulli(
            probs=probs
        )
        samples = existence_dist.sample()
        return samples, mask


class ARPermutationDecoder(nn.Module):
    """
    Pointer-style autoregressive decoder for root-to-leaf topological orders.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        self.hidden_dim = hidden_dim
        self.node_proj = nn.Linear(d_model, hidden_dim, bias=False, **factory_kwargs)
        self.state_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, **factory_kwargs)
        self.score_proj = nn.Linear(hidden_dim, 1, bias=False, **factory_kwargs)
        self.init_proj = nn.Linear(d_model, hidden_dim, bias=False, **factory_kwargs)
        self.gru = nn.GRUCell(d_model, hidden_dim, bias=False, **factory_kwargs)

    def _initial_state(self, q_rep: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.init_proj(q_rep.mean(dim=1)))

    def _step_logits(
        self,
        q_rep: torch.Tensor,
        state: torch.Tensor,
        selected: torch.Tensor,
        valid_nodes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scores = self.score_proj(
            torch.tanh(self.node_proj(q_rep) + self.state_proj(state).unsqueeze(1))
        ).squeeze(-1)
        scores = scores.masked_fill(selected, -torch.inf)
        if valid_nodes is not None:
            scores = scores.masked_fill(~valid_nodes, -torch.inf)
        return scores

    def log_prob(
        self,
        q_rep: torch.Tensor,
        orders: torch.Tensor,
        valid_nodes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Teacher-forced log P(order | X).

        Args:
            q_rep: [batch_size, num_nodes, d_model]
            orders: [num_orders, batch_size, num_nodes], root-to-leaf.
            valid_nodes: optional bool tensor [batch_size, num_nodes].

        Returns:
            log probabilities with shape [num_orders, batch_size].
        """
        num_orders, batch_size, order_len = orders.shape
        q_rep_rep = q_rep.repeat_interleave(num_orders, dim=0)
        flat_orders = orders.transpose(0, 1).contiguous().view(batch_size * num_orders, order_len)
        if valid_nodes is not None:
            valid_nodes = valid_nodes.repeat_interleave(num_orders, dim=0)

        selected = torch.zeros(
            (batch_size * num_orders, q_rep.size(1)),
            dtype=torch.bool,
            device=q_rep.device,
        )
        state = self._initial_state(q_rep_rep)
        log_prob = torch.zeros(batch_size * num_orders, device=q_rep.device, dtype=q_rep.dtype)
        batch_idx = torch.arange(batch_size * num_orders, device=q_rep.device)

        for t in range(order_len):
            logits = self._step_logits(q_rep_rep, state, selected, valid_nodes=valid_nodes)
            log_step = torch.log_softmax(logits.float(), dim=-1).to(q_rep.dtype)
            chosen = flat_orders[:, t].long()
            log_prob = log_prob + log_step[batch_idx, chosen]
            selected = selected.clone()
            selected[batch_idx, chosen] = True
            state = self.gru(q_rep_rep[batch_idx, chosen], state)

        return log_prob.view(batch_size, num_orders).transpose(0, 1)

    def sample(
        self,
        q_rep: torch.Tensor,
        num_samples: int,
        valid_nodes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample root-to-leaf orders from the autoregressive decoder.

        Returns:
            orders: [num_samples, batch_size, num_nodes]
            log_prob: [num_samples, batch_size]
        """
        batch_size, num_nodes, _ = q_rep.shape
        q_rep_rep = q_rep.repeat_interleave(num_samples, dim=0)
        if valid_nodes is not None:
            valid_nodes = valid_nodes.repeat_interleave(num_samples, dim=0)

        selected = torch.zeros(
            (batch_size * num_samples, num_nodes),
            dtype=torch.bool,
            device=q_rep.device,
        )
        state = self._initial_state(q_rep_rep)
        orders = torch.empty(
            (batch_size * num_samples, num_nodes),
            dtype=torch.long,
            device=q_rep.device,
        )
        log_prob = torch.zeros(batch_size * num_samples, device=q_rep.device, dtype=q_rep.dtype)
        batch_idx = torch.arange(batch_size * num_samples, device=q_rep.device)

        for t in range(num_nodes):
            logits = self._step_logits(q_rep_rep, state, selected, valid_nodes=valid_nodes)
            dist = torch.distributions.Categorical(logits=logits.float())
            chosen = dist.sample()
            orders[:, t] = chosen
            log_prob = log_prob + dist.log_prob(chosen).to(q_rep.dtype)
            selected = selected.clone()
            selected[batch_idx, chosen] = True
            state = self.gru(q_rep_rep[batch_idx, chosen], state)

        orders = orders.view(batch_size, num_samples, num_nodes).transpose(0, 1).contiguous()
        log_prob = log_prob.view(batch_size, num_samples).transpose(0, 1).contiguous()
        return orders, log_prob


class CausalProbabilisticARDecoder(CausalTNPEncoder):
    """
    Probabilistic decoder with autoregressive P(Q|X) and bak-style L/mask.
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
        num_topo_order_samples: int = 8,
        ar_hidden_dim: Optional[int] = None,
        Q_before_L=True,
        device=None,
        dtype=None,
        mlp_use_bias: bool = False,
        **kwargs,
    ):
        super(CausalProbabilisticARDecoder, self).__init__(
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
        self.n_perm_samples = n_perm_samples
        self.sinkhorn_iter = sinkhorn_iter
        self.num_topo_order_samples = num_topo_order_samples
        self.Q_before_L = Q_before_L
        print(f"Using {num_layers_decoder // 2} AR decoder layers.")
        self.decoder_L = nn.TransformerDecoder(
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
            num_layers=num_layers_decoder // 2,
        )
        self.decoder_Q = nn.TransformerDecoder(
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
            num_layers=num_layers_decoder // 2,
        )
        self.L_param = CausalAdjacencyMatrix(
            nhead=nhead,
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.ar_perm_decoder = ARPermutationDecoder(
            d_model=d_model,
            hidden_dim=ar_hidden_dim,
            device=device,
            dtype=dtype,
        )

    def decode(self, representation, mask: Optional[torch.Tensor] = None):
        if not self.Q_before_L:
            L_rep = self.decoder_L(representation, memory=None, tgt_key_padding_mask=mask)
            Q_rep = self.decoder_Q(L_rep, memory=None, tgt_key_padding_mask=mask)
            L_param = self.L_param(L_rep, padding_mask=mask)
        else:
            Q_rep = self.decoder_Q(representation, memory=None, tgt_key_padding_mask=mask)
            L_rep = self.decoder_L(Q_rep, memory=None, tgt_key_padding_mask=mask)
            L_param = self.L_param(L_rep, padding_mask=mask)
        L_param = (L_param + L_param.transpose(1, 2)) / 2
        return L_param, Q_rep

    def _encode_decode(self, target_data, mask: Optional[torch.Tensor] = None):
        if self.num_nodes != target_data.size(-1):
            raise ValueError("Number of nodes in the input data should be equal to num_nodes.")
        if target_data.dim() == 3:
            target_data = target_data.unsqueeze(-1)
        representation = self.encode(target_data=target_data, mask=mask).squeeze(2)
        decoder_mask = mask[:, 0, :] if mask is not None else None
        L_param, Q_rep = self.decode(representation=representation, mask=decoder_mask)
        return L_param, Q_rep, decoder_mask

    def _lower_mask(self, size: int, device, dtype):
        return torch.tril(
            torch.ones((self.num_nodes, self.num_nodes), device=device, dtype=torch.float32),
            diagonal=-1,
        ).to(dtype)[:size, :size]

    def _graph_log_prob(self, edge_probs, perm, target):
        perm_inv = torch.transpose(perm, 3, 2)
        tri_mask = self._lower_mask(target.size(-1), perm.device, perm.dtype)
        all_masks = torch.einsum("sbij,jk,sbkl->sbil", perm, tri_mask, perm_inv)
        probs = edge_probs[None] * all_masks
        probs = probs.clamp(1e-6, 1 - 1e-6)
        existence_dist = torch.distributions.Bernoulli(probs=probs)
        log_prob = existence_dist.log_prob(target[None])
        return log_prob.flatten(start_dim=2).sum(dim=-1), probs

    def forward(self, target_data, graph, mask: Optional[torch.Tensor] = None, is_training=True):
        L_param, Q_rep, decoder_mask = self._encode_decode(target_data=target_data, mask=mask)
        edge_probs = torch.sigmoid(L_param)
        valid_nodes = (decoder_mask != -float("inf")) if decoder_mask is not None else None

        if graph is None:
            orders, log_p_q = self.ar_perm_decoder.sample(
                Q_rep,
                num_samples=self.n_perm_samples,
                valid_nodes=valid_nodes,
            )
            perm = orders_to_bak_permutation(orders, num_nodes=Q_rep.size(1), dtype=edge_probs.dtype)
            _, probs = self._graph_log_prob(edge_probs=edge_probs, perm=perm, target=torch.zeros_like(edge_probs))
            return probs

        orders = sample_topological_orders(graph, self.num_topo_order_samples)
        log_p_q = self.ar_perm_decoder.log_prob(Q_rep, orders, valid_nodes=valid_nodes)
        perm = orders_to_bak_permutation(orders, num_nodes=Q_rep.size(1), dtype=edge_probs.dtype)
        log_p_g, _ = self._graph_log_prob(edge_probs=edge_probs, perm=perm, target=graph)
        return {
            "scores": log_p_q + log_p_g,
            "orders": orders,
            "log_p_q": log_p_q,
            "log_p_g": log_p_g,
        }

    def calculate_loss(self, output, target):
        if isinstance(output, dict):
            scores = output["scores"]
            loss = -(torch.logsumexp(scores.float(), dim=0) - math.log(scores.size(0)))
            return loss.to(scores.dtype)

        probs = output.contiguous().view(output.size(0), output.size(1), -1)
        target_graph = target.reshape(target.size(0), -1)
        existence_dist = torch.distributions.Bernoulli(probs=probs.clamp(1e-6, 1 - 1e-6))
        log_prob = existence_dist.log_prob(target_graph[None])
        log_prob_sum = torch.logsumexp(log_prob, dim=0) - math.log(log_prob.size(0))
        loss_per_edge = -log_prob_sum
        return loss_per_edge.mean(dim=1)

    def sample(self, target_data: torch.Tensor, num_samples: int, mask: Optional[torch.Tensor] = None):
        old_n_perm_samples = self.n_perm_samples
        self.n_perm_samples = num_samples
        probs = self.forward(target_data, graph=None, is_training=False, mask=mask)
        self.n_perm_samples = old_n_perm_samples
        existence_dist = torch.distributions.Bernoulli(probs=probs)
        samples = existence_dist.sample()
        return samples, mask
