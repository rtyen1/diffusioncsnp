"""
Utilities for sampling and converting topological orders.
"""

from __future__ import annotations

from typing import List, Optional

import torch


def random_kahn_topological_sort(adj: torch.Tensor) -> List[int]:
    """
    Sample one root-to-leaf topological order from a DAG using randomized Kahn.

    Args:
        adj: Tensor of shape [num_nodes, num_nodes], where adj[i, j] = 1 means i -> j.

    Returns:
        A list of node indices in root-to-leaf topological order.
    """
    graph = (adj.detach().cpu() > 0.5).to(torch.int64)
    num_nodes = graph.size(0)
    indegree = graph.sum(dim=0).tolist()
    available = [i for i in range(num_nodes) if indegree[i] == 0]
    order: List[int] = []

    while available:
        idx = int(torch.randint(len(available), (1,)).item())
        node = available.pop(idx)
        order.append(node)
        children = torch.nonzero(graph[node], as_tuple=False).flatten().tolist()
        for child in children:
            indegree[child] -= 1
            if indegree[child] == 0:
                available.append(child)

    if len(order) != num_nodes:
        raise ValueError("Graph is cyclic; cannot sample a topological order.")
    return order


def priority_kahn_topological_sort(adj: torch.Tensor, priority: torch.Tensor) -> List[int]:
    """
    Produce a unique root-to-leaf topological order using priority tie-breaking.

    Args:
        adj: Tensor of shape [num_nodes, num_nodes], where adj[i, j] = 1 means i -> j.
        priority: Tensor of shape [num_nodes]. Smaller values are selected earlier
            among currently available source nodes.

    Returns:
        A list of node indices in root-to-leaf topological order.
    """
    graph = (adj.detach().cpu() > 0.5).to(torch.int64)
    priority_cpu = priority.detach().cpu()
    num_nodes = graph.size(0)
    if priority_cpu.numel() != num_nodes:
        raise ValueError("priority must have one value per node.")

    indegree = graph.sum(dim=0).tolist()
    available = [i for i in range(num_nodes) if indegree[i] == 0]
    order: List[int] = []

    while available:
        node = min(available, key=lambda i: (float(priority_cpu[i]), i))
        available.remove(node)
        order.append(node)
        children = torch.nonzero(graph[node], as_tuple=False).flatten().tolist()
        for child in children:
            indegree[child] -= 1
            if indegree[child] == 0:
                available.append(child)

    if len(order) != num_nodes:
        raise ValueError("Graph is cyclic; cannot sample a topological order.")
    return order


def sample_topological_orders(target: torch.Tensor, num_orders: int) -> torch.Tensor:
    """
    Sample K topological orders for each graph in a batch.

    Args:
        target: Tensor of shape [batch_size, num_nodes, num_nodes].
        num_orders: Number of orders to sample per graph.

    Returns:
        Tensor of shape [num_orders, batch_size, num_nodes], root-to-leaf.
    """
    if target.dim() != 3:
        raise ValueError("target must have shape [batch_size, num_nodes, num_nodes].")
    batch_size, num_nodes, _ = target.shape
    orders = torch.empty(
        (num_orders, batch_size, num_nodes),
        dtype=torch.long,
        device=target.device,
    )
    for b in range(batch_size):
        for k in range(num_orders):
            orders[k, b] = torch.tensor(
                random_kahn_topological_sort(target[b]),
                dtype=torch.long,
                device=target.device,
            )
    return orders


def orders_to_bak_permutation(
    orders: torch.Tensor,
    num_nodes: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Convert root-to-leaf orders to bak-compatible permutation matrices.

    In the bak model, the strict lower-triangular mask means larger positions
    are closer to roots. Therefore a root-to-leaf order is mapped so that the
    first/root node gets position D - 1 and the last/leaf node gets position 0.

    Args:
        orders: Tensor of shape [num_orders, batch_size, num_nodes].
        num_nodes: Number of nodes. Defaults to orders.size(-1).
        dtype: Output dtype. Defaults to float32.

    Returns:
        Tensor of shape [num_orders, batch_size, num_nodes, num_nodes], where
        perm[s, b, node, position] = 1.
    """
    if orders.dim() != 3:
        raise ValueError("orders must have shape [num_orders, batch_size, num_nodes].")
    num_orders, batch_size, order_len = orders.shape
    if num_nodes is None:
        num_nodes = order_len
    if dtype is None:
        dtype = torch.float32

    perm = torch.zeros(
        (num_orders, batch_size, num_nodes, num_nodes),
        dtype=dtype,
        device=orders.device,
    )
    positions = torch.arange(order_len - 1, -1, -1, device=orders.device)
    positions = positions.view(1, 1, order_len).expand(num_orders, batch_size, order_len)
    sample_idx = torch.arange(num_orders, device=orders.device).view(num_orders, 1, 1)
    sample_idx = sample_idx.expand(num_orders, batch_size, order_len)
    batch_idx = torch.arange(batch_size, device=orders.device).view(1, batch_size, 1)
    batch_idx = batch_idx.expand(num_orders, batch_size, order_len)
    perm[sample_idx, batch_idx, orders.long(), positions] = 1
    return perm
