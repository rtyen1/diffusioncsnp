"""Distribution errors against the exact simulator posterior (float64)."""
from __future__ import annotations

import csv
import io

import numpy as np

from test_code.posterior_benchmark.generate_training_data import atomic_bytes


def logsumexp(x, axis=-1, keepdims=False):
    maximum = np.max(x, axis=axis, keepdims=True)
    value = maximum + np.log(np.exp(x - maximum).sum(axis=axis, keepdims=True))
    return value if keepdims else np.squeeze(value, axis=axis)


def distribution_metrics(true_log, pred_log, classes, mec_ids):
    true_log, pred_log = np.asarray(true_log, dtype=np.float64), np.asarray(pred_log, dtype=np.float64)
    if true_log.shape != pred_log.shape or true_log.ndim != 2 or true_log.shape[-1] != 25:
        raise ValueError("Expected matching [datasets,25] log posteriors")
    if not np.isfinite(true_log).all() or not np.isfinite(pred_log).all():
        raise ValueError("Nonfinite log posterior")
    for name, value in (("true", true_log), ("predicted", pred_log)):
        if not np.allclose(logsumexp(value), 0., atol=2e-5, rtol=0):
            raise ValueError(f"Unnormalized {name} graph distribution")
    # Only numerical normalization, never a graph mask / threshold / prior correction.
    true_log = true_log - logsumexp(true_log, keepdims=True)
    pred_log = pred_log - logsumexp(pred_log, keepdims=True)
    p, q = np.exp(true_log), np.exp(pred_log)
    middle = np.logaddexp(true_log, pred_log) - np.log(2.)
    pm = np.stack([logsumexp(true_log[:, mec_ids == m]) for m in range(1, 12)], axis=1)
    qm = np.stack([logsumexp(pred_log[:, mec_ids == m]) for m in range(1, 12)], axis=1)
    rows = np.arange(len(p))
    true_mec = mec_ids[np.asarray(classes, dtype=int)] - 1
    within_tv = np.empty(len(p))
    for m in range(1, 12):
        selected = np.flatnonzero(true_mec == m - 1)
        mask = mec_ids == m
        if len(selected):
            pp = np.exp(true_log[selected][:, mask] - pm[selected, m - 1, None])
            qq = np.exp(pred_log[selected][:, mask] - qm[selected, m - 1, None])
            within_tv[selected] = .5 * np.abs(pp - qq).sum(axis=-1)
    return {
        "kl": np.maximum(0., (p * (true_log - pred_log)).sum(axis=-1)),
        "tv": .5 * np.abs(p - q).sum(axis=-1),
        "js": np.maximum(0., .5 * (p * (true_log - middle) + q * (pred_log - middle)).sum(axis=-1)),
        "graph_nll": -pred_log[rows, classes],
        "true_entropy": -(p * true_log).sum(axis=-1),
        "pred_entropy": -(q * pred_log).sum(axis=-1),
        "mec_kl": np.maximum(0., (np.exp(pm) * (pm - qm)).sum(axis=-1)),
        "true_mec_mass": np.exp(pm[rows, true_mec]),
        "pred_mec_mass": np.exp(qm[rows, true_mec]),
        "within_true_mec_tv": within_tv,
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty metric table")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, buffer.getvalue().encode())


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["method"], int(row["training_seed"]), int(row["sample_size"])), []).append(row)
    result = []
    keys = [key for key in rows[0] if key not in {
        "method", "training_seed", "sample_size", "dataset_seed", "class_index", "mec_id"}]
    for (method, seed, n), group in sorted(grouped.items()):
        record = {"method": method, "training_seed": seed, "sample_size": n, "datasets": len(group)}
        for key in keys:
            values = np.asarray([row[key] for row in group], dtype=np.float64)
            record[f"mean_{key}"] = float(np.mean(values))
            record[f"std_{key}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
        result.append(record)
    return result
