#!/usr/bin/env python3
"""Shared-prior d=3 training data with exactly enumerated graph posteriors.

Only NumPy and h5py are required. The 25 fixed SEMs are read from the bundled
LaTeX specification and checked against its 11 declared MEC covariances.
No normalization, iSCM operation, parameter resampling or graph augmentation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import h5py
import numpy as np

VARIABLES = ("A", "B", "C")
SPLITS = ("train", "val", "test")
FEATURES = ("n", "S_AA", "S_BB", "S_CC", "S_AB", "S_AC", "S_BC")
META_DTYPE = np.dtype([("dataset_seed", "<u8"), ("graph_id", "u1"),
    ("class_index", "u1"), ("mec_id", "u1"), ("sample_size", "<u4")])


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def atomic_bytes(path, payload):
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def topological_order(graph):
    indegree = graph.sum(axis=0).astype(int)
    ready = list(np.flatnonzero(indegree == 0))
    order = []
    while ready:
        node = int(ready.pop())
        order.append(node)
        for child in np.flatnonzero(graph[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(int(child))
    if len(order) != 3:
        raise ValueError("Cyclic or non-three-node graph")
    return np.asarray(order, dtype=np.int8)


def mec_signature(graph):
    skeleton = (graph | graph.T).astype(bool)
    colliders = []
    for target in range(3):
        for left, right in itertools.combinations(np.flatnonzero(graph[:, target]), 2):
            if not skeleton[left, right]:
                colliders.append((int(left), target, int(right)))
    return tuple(skeleton.ravel()), tuple(sorted(colliders))


@dataclass
class Catalogue:
    source: Path
    source_sha256: str
    coefficients: np.ndarray       # B[g,i,j] means coefficient on i -> j
    noise_variances: np.ndarray
    graphs: np.ndarray
    orders: np.ndarray
    mec_ids: np.ndarray            # 1..11, indexed by class 0..24
    mec_covariances: np.ndarray
    cholesky: np.ndarray
    precisions: np.ndarray
    log_determinants: np.ndarray
    sem_factors: np.ndarray        # F F^T = covariance, X = Z F^T

    @property
    def prior(self):
        return np.full(25, 1. / 25., dtype=np.float64)


def load_catalogue(path):
    """Strict parser for the supplied fixed-SEM document, not arbitrary LaTeX."""
    path = Path(path).resolve()
    text = path.read_text()
    if not re.search(r"\\mu=\(0,\s*0,\s*0\)", text):
        raise ValueError("This experiment requires the specified known mean (0,0,0)")
    covariances = {}
    covariance_pattern = r"\\Sigma_(?:\{(\d+)\}|(\d+))=\\begin\{pmatrix\}(.+?)\\end\{pmatrix\}"
    for match in re.finditer(covariance_pattern, text, re.S):
        mec = int(match[1] or match[2])
        if mec in covariances:
            raise ValueError(f"Duplicate covariance for MEC {mec}")
        covariances[mec] = np.asarray([
            [float(value.strip()) for value in row.split("&")]
            for row in match[3].split(r"\\") if row.strip()], dtype=np.float64)
    if set(covariances) != set(range(1, 12)):
        raise ValueError("Expected exactly the 11 declared MEC covariances")
    models = {}
    dag_pattern = r"\\noindent\\textbf\{DAG \\#(\d+):(.+?)\(MEC (\d+)\)\}\s*\\begin\{align\*\}(.*?)\\end\{align\*\}"
    row_pattern = (r"([ABC])\s*&=\s*(.*?),\s*&\s*\\varepsilon_([ABC])\s*"
                   r"\\sim\s*\\mathcal\{N\}\(0,\s*([0-9.]+)\)")
    term_pattern = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))([ABC])"
    for match in re.finditer(dag_pattern, text, re.S):
        graph_id, mec = int(match[1]), int(match[3])
        if graph_id in models:
            raise ValueError(f"Duplicate graph ID {graph_id}")
        rows = re.findall(row_pattern, match[4])
        if len(rows) != 3 or {row[0] for row in rows} != set(VARIABLES):
            raise ValueError(f"Cannot parse all three equations for DAG {graph_id}")
        coefficients, variances = np.zeros((3, 3)), np.zeros(3)
        for target, rhs, noise, variance in rows:
            if target != noise:
                raise ValueError("Noise variable does not match structural equation")
            j = VARIABLES.index(target)
            variances[j] = float(variance)
            expression = re.sub(r"\s+", "", rhs)
            epsilon = r"\varepsilon_" + target
            if expression.count(epsilon) != 1:
                raise ValueError("Expected one own-node additive Gaussian noise")
            expression = expression.replace(epsilon, "")
            for weight, parent in re.findall(term_pattern, expression):
                i = VARIABLES.index(parent)
                if i == j or coefficients[i, j] != 0:
                    raise ValueError("Duplicate parent term or self-loop")
                coefficients[i, j] = float(weight)
            if re.sub(term_pattern, "", expression).strip("+-"):
                raise ValueError(f"Unsupported equation for DAG {graph_id}: {rhs}")
        graph = (coefficients != 0).astype(np.int8)
        header_edges = {(VARIABLES.index(a), VARIABLES.index(b))
                        for a, b in re.findall(r"([ABC])->([ABC])", match[2])}
        if header_edges != set(zip(*np.nonzero(graph))):
            raise ValueError(f"Header / structural equation edge mismatch at DAG {graph_id}")
        order = topological_order(graph)
        if not np.isfinite(variances).all() or not (variances > 0).all():
            raise ValueError("Structural variances must be finite and positive")
        transform = np.linalg.solve(np.eye(3) - coefficients.T, np.eye(3))
        factor = transform @ np.diag(np.sqrt(variances))
        derived = factor @ factor.T
        if mec not in covariances or not np.allclose(derived, covariances[mec], atol=1e-12, rtol=1e-12):
            raise ValueError(f"SEM / declared covariance mismatch at DAG {graph_id}")
        models[graph_id] = (coefficients, variances, graph, order, mec, factor)
    if set(models) != set(range(1, 26)):
        raise ValueError("Expected exactly graph IDs 1..25")
    values = [models[index] for index in range(1, 26)]
    graphs = np.stack([value[2] for value in values])
    if len({tuple(graph.ravel()) for graph in graphs}) != 25:
        raise ValueError("Duplicate DAGs")
    all_dags = set()
    edges = [(i, j) for i in range(3) for j in range(3) if i != j]
    for bits in itertools.product((0, 1), repeat=6):
        graph = np.zeros((3, 3), dtype=np.int8)
        for edge, bit in zip(edges, bits):
            graph[edge] = bit
        try:
            topological_order(graph)
        except ValueError:
            continue
        all_dags.add(tuple(graph.ravel()))
    if {tuple(graph.ravel()) for graph in graphs} != all_dags:
        raise ValueError("Catalogue is not an enumeration of all three-node DAGs")
    for left in values:
        for right in values:
            if (left[4] == right[4]) != (mec_signature(left[2]) == mec_signature(right[2])):
                raise ValueError("Declared MECs disagree with skeleton/v-structure equivalence")
    covariance_array = np.stack([covariances[mec] for mec in range(1, 12)])
    if covariance_array.shape != (11, 3, 3) or not np.isfinite(covariance_array).all():
        raise ValueError("Invalid covariance array")
    if not np.array_equal(covariance_array, covariance_array.transpose(0, 2, 1)):
        raise ValueError("Covariances must be symmetric")
    cholesky = np.linalg.cholesky(covariance_array)
    return Catalogue(path, file_hash(path), np.stack([v[0] for v in values]),
        np.stack([v[1] for v in values]), graphs, np.stack([v[3] for v in values]),
        np.array([v[4] for v in values], dtype=np.uint8), covariance_array, cholesky,
        np.linalg.inv(covariance_array), 2*np.log(np.diagonal(cholesky, axis1=1, axis2=2)).sum(axis=1),
        np.stack([v[5] for v in values]))


def sample_sem(catalogue, class_index, standard_noise):
    """Original parent-to-child equations with the same external standard noise."""
    z = np.asarray(standard_noise, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 3:
        raise ValueError("Expected [observations,3] independent standard noises")
    noise = z * np.sqrt(catalogue.noise_variances[class_index])
    coefficients = catalogue.coefficients[class_index]
    data = np.empty_like(z)
    for node in catalogue.orders[class_index]:
        parents = np.flatnonzero(coefficients[:, node])
        data[:, node] = noise[:, node]
        if parents.size:
            data[:, node] += data[:, parents] @ coefficients[parents, node]
    return data


def check_sem_equivalence(catalogue):
    """Algebraic and shared-noise checks, not merely a covariance simulation."""
    z = np.random.default_rng(913).standard_normal((128, 3))
    max_error = 0.
    for index in range(25):
        sequential = sample_sem(catalogue, index, z)
        matrix_sem = z @ catalogue.sem_factors[index].T
        np.testing.assert_allclose(sequential, matrix_sem, atol=2e-14, rtol=2e-14)
        # Different Gaussian factors do not produce identical observations
        # from the SAME Z. Couple them using an orthogonal noise rotation.
        lower = catalogue.cholesky[int(catalogue.mec_ids[index]) - 1]
        rotation = np.linalg.solve(lower, catalogue.sem_factors[index]).T
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=2e-14, rtol=2e-14)
        covariance_sample = (z @ rotation) @ lower.T
        np.testing.assert_allclose(covariance_sample, sequential, atol=2e-14, rtol=2e-14)
        max_error = max(max_error, float(np.max(np.abs(sequential - matrix_sem))))
    return {"models_checked": 25, "mecs_checked": 11,
            "max_shared_noise_sem_matrix_error": max_error,
            "equivalence": "same observational Gaussian law; Cholesky is NOT the causal graph"}


def sufficient_statistics(data):
    x = np.asarray(data, dtype=np.float64)
    if x.ndim < 2 or x.shape[-1] != 3 or not np.isfinite(x).all():
        raise ValueError("Expected finite [...,observations,3] raw data")
    scatter = np.einsum("...ni,...nj->...ij", x, x, optimize=False)
    return scatter, features_from_scatter(scatter, x.shape[-2])


def features_from_scatter(scatter, n):
    return np.stack([np.full(scatter.shape[:-2], n, dtype=np.float64),
        scatter[..., 0, 0], scatter[..., 1, 1], scatter[..., 2, 2],
        scatter[..., 0, 1], scatter[..., 0, 2], scatter[..., 1, 2]], axis=-1)


def exact_posterior(catalogue, scatter, n):
    """Known mean=0, fixed SEM likelihood, uniform DAG prior, all 25 graphs."""
    scatter = np.asarray(scatter, dtype=np.float64)
    if scatter.shape[-2:] != (3, 3) or not np.isfinite(scatter).all() or n < 0:
        raise ValueError("Invalid raw scatter or observation count")
    trace = np.einsum("...ij,cji->...c", scatter, catalogue.precisions, optimize=False)
    log_likelihood = -.5 * (trace + n*catalogue.log_determinants + n*3*math.log(2*math.pi))
    log_joint = log_likelihood[..., catalogue.mec_ids.astype(int)-1] + np.log(catalogue.prior)
    maximum = log_joint.max(axis=-1, keepdims=True)
    log_evidence = maximum + np.log(np.exp(log_joint-maximum).sum(axis=-1, keepdims=True))
    log_posterior = log_joint - log_evidence
    posterior = np.exp(log_posterior)
    mec_posterior = np.stack([posterior[..., catalogue.mec_ids == mec].sum(axis=-1)
                              for mec in range(1, 12)], axis=-1)
    return posterior, log_posterior, mec_posterior, log_evidence[..., 0]


def analytic_softmax_parameters(catalogue):
    omega = catalogue.precisions[catalogue.mec_ids.astype(int)-1]
    weights = np.stack([-.5*catalogue.log_determinants[catalogue.mec_ids.astype(int)-1],
        -.5*omega[:, 0, 0], -.5*omega[:, 1, 1], -.5*omega[:, 2, 2],
        -omega[:, 0, 1], -omega[:, 0, 2], -omega[:, 1, 2]], axis=-1)
    return weights, np.log(catalogue.prior)


def catalogue_document(catalogue):
    weights, bias = analytic_softmax_parameters(catalogue)
    return {"source_file": catalogue.source.name, "source_sha256": catalogue.source_sha256,
        "variables": list(VARIABLES), "mean": [0., 0., 0.],
        "label_convention": "B[i,j] and label[i,j] refer to i -> j",
        "class_index": "0..24; graph_id = class_index + 1; IDs follow the LaTeX",
        "graph_prior": catalogue.prior.tolist(),
        "mec_prior": [float(catalogue.prior[catalogue.mec_ids == mec].sum()) for mec in range(1, 12)],
        "models": [{"graph_id": i+1, "class_index": i, "mec_id": int(catalogue.mec_ids[i]),
                    "coefficients": catalogue.coefficients[i].tolist(),
                    "noise_variances": catalogue.noise_variances[i].tolist(),
                    "label": catalogue.graphs[i].tolist(), "topological_order": catalogue.orders[i].tolist(),
                    "covariance": catalogue.mec_covariances[int(catalogue.mec_ids[i])-1].tolist()}
                   for i in range(25)],
        "sufficient_statistics_fields": list(FEATURES),
        "analytic_softmax_weights": weights.tolist(), "analytic_softmax_bias": bias.tolist()}


def seed_for(seed, split, n, index, stream):
    return int(np.random.SeedSequence([seed, SPLITS.index(split), n, index, stream])
               .generate_state(1, dtype=np.uint64)[0])


def shard_classes(config, job):
    # Every shard is balanced over DAGs, NOT over MECs. Shuffle independently
    # of the observational-noise streams and hide file-row/class ordering.
    labels = np.repeat(np.arange(25, dtype=np.uint8), job["count"] // 25)
    rng = np.random.default_rng(seed_for(config["seed"], job["split"], job["n"], job["shard"], 0))
    rng.shuffle(labels)
    return labels


def sample_batch(catalogue, classes, seeds, n, method="covariance"):
    z = np.stack([np.random.default_rng(int(seed)).standard_normal((n, 3)) for seed in seeds])
    if method == "sem":
        return np.stack([sample_sem(catalogue, int(index), noise) for index, noise in zip(classes, z)])
    if method != "covariance":
        raise ValueError(f"Unknown sampling method: {method}")
    factors = catalogue.cholesky[catalogue.mec_ids[classes].astype(int)-1]
    return np.einsum("bni,bji->bnj", z, factors, optimize=False)


def expected_metadata(config, job, catalogue):
    result = np.zeros(job["count"], dtype=META_DTYPE)
    result["class_index"] = shard_classes(config, job)
    result["graph_id"] = result["class_index"] + 1
    result["mec_id"] = catalogue.mec_ids[result["class_index"]]
    result["sample_size"] = job["n"]
    result["dataset_seed"] = [seed_for(config["seed"], job["split"], job["n"], job["start"]+i, 1)
                               for i in range(job["count"])]
    return result


def stored_shapes(count, n):
    return {"data": ((count, n, 3), "f8"), "label": ((count, 3, 3), "i1"),
        "graph_id": ((count,), "u1"), "class_index": ((count,), "u1"), "mec_id": ((count,), "u1"),
        "dataset_seed": ((count,), "u8"), "scatter": ((count, 3, 3), "f8"),
        "sufficient_statistics": ((count, 7), "f8"), "posterior": ((count, 25), "f8"),
        "log_posterior": ((count, 25), "f8"), "mec_posterior": ((count, 11), "f8"),
        "log_evidence": ((count,), "f8"), "metadata": ((count,), META_DTYPE),
        "completed": ((count,), "?"), "data_sha256": ((count,), "S64")}


def validate_handle(handle, config, job, catalogue, batch_size):
    if handle.attrs.get("generation_config") != canonical(config):
        raise ValueError(f"Shard protocol mismatch: {handle.filename}")
    for field in ("split", "n", "shard", "start", "count"):
        if handle.attrs.get(field) != job[field]:
            raise ValueError(f"Shard location / metadata mismatch: {field}")
    for name, (shape, dtype) in stored_shapes(job["count"], job["n"]).items():
        if handle[name].shape != shape or handle[name].dtype != np.dtype(dtype):
            raise ValueError(f"Wrong HDF5 shape or dtype: {name}")
    metadata = expected_metadata(config, job, catalogue)
    completed = handle["completed"][:]
    for start in range(0, job["count"], batch_size):
        indices = np.flatnonzero(completed[start:start+batch_size]) + start
        if not len(indices):
            continue
        record = metadata[indices]
        np.testing.assert_array_equal(handle["metadata"][indices], record)
        for name in ("graph_id", "class_index", "mec_id", "dataset_seed"):
            np.testing.assert_array_equal(handle[name][indices], record[name])
        np.testing.assert_array_equal(handle["label"][indices], catalogue.graphs[record["class_index"]])
        data = handle["data"][indices]
        scatter, features = sufficient_statistics(data)
        p, lp, mec, evidence = exact_posterior(catalogue, scatter, job["n"])
        for name, expected in (("scatter", scatter), ("sufficient_statistics", features),
                               ("posterior", p), ("log_posterior", lp),
                               ("mec_posterior", mec), ("log_evidence", evidence)):
            np.testing.assert_allclose(handle[name][indices], expected, rtol=1e-12, atol=1e-12,
                                       err_msg=f"Incorrect {name} in {handle.filename}")
        np.testing.assert_allclose(handle["posterior"][indices].sum(axis=-1), 1., rtol=0, atol=1e-12)
        hashes = handle["data_sha256"][indices]
        if any(data_hash(array).encode() != checksum for array, checksum in zip(data, hashes)):
            raise ValueError(f"Observation checksum mismatch: {handle.filename}")
    return completed


def write_shard(config, job, catalogue, batch_size):
    path = Path(job["path"])
    if path.exists():
        with h5py.File(path, "r") as handle:
            if not validate_handle(handle, config, job, catalogue, batch_size).all():
                raise ValueError(f"Incomplete final shard: {path}")
        print(f"[VERIFIED/SKIP] {path}", flush=True)
        return
    partial = path.with_name(f".{path.stem}.partial.hdf5")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not partial.exists():
        fd, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        try:
            with h5py.File(name, "w") as handle:
                for key, (shape, dtype) in stored_shapes(job["count"], job["n"]).items():
                    kwargs = {"chunks": (min(job["count"], 16), min(job["n"], 256), 3)} if key == "data" else {}
                    handle.create_dataset(key, shape=shape, dtype=dtype, **kwargs)
                handle.attrs.update({key: job[key] for key in ("split", "n", "shard", "start", "count")})
                handle.attrs.update(generation_config=canonical(config), n_nodes=3, num_samples=job["n"],
                    columns_json=canonical(VARIABLES), label_convention="label[i, j] = 1 means i -> j",
                    input_normalization="none", mean_known=True, fixed_sem_parameters=True,
                    feature_names_json=canonical(FEATURES))
            os.replace(name, partial)
        finally:
            if os.path.exists(name):
                os.unlink(name)
    metadata = expected_metadata(config, job, catalogue)
    started = time.monotonic()
    with h5py.File(partial, "r+") as handle:
        done = validate_handle(handle, config, job, catalogue, batch_size)
        pending = np.flatnonzero(~done)
        for begin in range(0, len(pending), batch_size):
            indices = pending[begin:begin+batch_size]
            records = metadata[indices]
            data = sample_batch(catalogue, records["class_index"], records["dataset_seed"],
                                job["n"], config["sampling_method"])
            scatter, features = sufficient_statistics(data)
            p, lp, mec, evidence = exact_posterior(catalogue, scatter, job["n"])
            payload = {"data": data, "label": catalogue.graphs[records["class_index"]],
                "scatter": scatter, "sufficient_statistics": features, "posterior": p,
                "log_posterior": lp, "mec_posterior": mec, "log_evidence": evidence,
                "metadata": records, "data_sha256": np.array([data_hash(row).encode() for row in data])}
            payload.update({name: records[name] for name in ("graph_id", "class_index", "mec_id", "dataset_seed")})
            for key, values in payload.items():
                handle[key][indices] = values
            handle.flush()
            handle["completed"][indices] = True
            handle.flush()
        if not validate_handle(handle, config, job, catalogue, batch_size).all():
            raise ValueError(f"Unfinished shard: {partial}")
    os.replace(partial, path)
    print(f"[COMPLETE] {path} datasets={job['count']} seconds={time.monotonic()-started:.2f}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sem-file", type=Path, default=Path(__file__).with_name("all_25_sems.tex"))
    parser.add_argument("--sample-sizes", default="20,100,500")
    parser.add_argument("--train-per-size", type=int, default=50_000)
    parser.add_argument("--val-per-size", type=int, default=2_500)
    parser.add_argument("--test-per-size", type=int, default=5_000)
    parser.add_argument("--datasets-per-file", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--sampling-method", choices=("covariance", "sem"), default="covariance")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    args.sem_file = args.sem_file.expanduser().resolve()
    try:
        args.sample_sizes = sorted(int(part.strip()) for part in args.sample_sizes.split(","))
    except ValueError:
        parser.error("Sample sizes must be comma-separated integers")
    if not args.sample_sizes or min(args.sample_sizes) < 2 or len(set(args.sample_sizes)) != len(args.sample_sizes):
        parser.error("Use distinct sample sizes >= 2")
    for count in (args.train_per_size, args.val_per_size, args.test_per_size, args.datasets_per_file):
        if count < 25 or count % 25:
            parser.error("Split counts and datasets-per-file must be positive multiples of 25")
    if args.batch_size < 1 or args.seed < 0:
        parser.error("Batch size must be positive; seed must be nonnegative")
    return args


def run_config(args, catalogue):
    return {"format": "d3_fixed25_uniform_graph_prior_exact_posterior_v1",
        "catalogue_sha256": hashlib.sha256(canonical(catalogue_document(catalogue)).encode()).hexdigest(),
        "source_sha256": catalogue.source_sha256, "graph_prior": "uniform over 25 DAGs, not 11 MECs",
        "sample_sizes": args.sample_sizes, "counts_per_size": {
            "train": args.train_per_size, "val": args.val_per_size, "test": args.test_per_size},
        "datasets_per_file": args.datasets_per_file, "seed": args.seed,
        "sampling_method": args.sampling_method, "data_dtype": "float64", "posterior_dtype": "float64",
        "mean": [0., 0., 0.], "normalization": "none", "node_permutation_augmentation": False,
        "iid_observations_within_dataset": True,
        "balanced_graph_labels": "exactly balanced within each split/n/shard; shuffled rows",
        "dataset_seed": "SeedSequence([seed,split_code,n,global_dataset_index,1])->uint64",
        "class_shuffle_seed": "SeedSequence([seed,split_code,n,shard_index,0])->uint64",
        "split_codes": {split: i for i, split in enumerate(SPLITS)},
        "posterior": "normalized exact fixed-SEM Gaussian likelihood times 1/25; known mean zero, uncentered scatter",
        "numpy_version": np.__version__, "h5py_version": h5py.__version__}


def jobs_for(args):
    result = []
    for split in SPLITS:
        count = getattr(args, f"{split}_per_size")
        for n in args.sample_sizes:
            for shard, start in enumerate(range(0, count, args.datasets_per_file)):
                result.append({"split": split, "n": n, "shard": shard, "start": start,
                    "count": min(args.datasets_per_file, count-start),
                    "path": str(args.output_root / split / f"n{n}/d3_shard_{shard:05d}.hdf5")})
    return result


def final_report(args, config, catalogue, jobs, equivalence):
    records, all_seeds = [], set()
    for split in SPLITS:
        for n in args.sample_sizes:
            counts = np.zeros(25, dtype=np.int64)
            max_error = 0.
            for job in jobs:
                if job["split"] != split or job["n"] != n:
                    continue
                with h5py.File(job["path"], "r") as handle:
                    if not validate_handle(handle, config, job, catalogue, args.batch_size).all():
                        raise ValueError(f"Incomplete final shard: {job['path']}")
                    counts += np.bincount(handle["class_index"][:], minlength=25)
                    seeds = [int(value) for value in handle["dataset_seed"][:]]
                    if len(set(seeds)) != len(seeds) or all_seeds.intersection(seeds):
                        raise ValueError("Repeated dataset seed across splits/shards/sample sizes")
                    all_seeds.update(seeds)
                    for start in range(0, job["count"], args.batch_size):
                        p = handle["posterior"][start:start+args.batch_size]
                        max_error = max(max_error, float(np.abs(p.sum(axis=-1)-1).max()))
            expected = getattr(args, f"{split}_per_size")
            if not np.array_equal(counts, np.full(25, expected//25)):
                raise ValueError(f"Graph prior not exactly balanced for {split}/n{n}")
            records.append({"split": split, "sample_size": n, "datasets": int(counts.sum()),
                "datasets_per_graph": counts.tolist(), "max_posterior_sum_error": max_error})
    return {"status": "COMPLETE", "graph_count": 25, "mec_count": 11,
        "total_datasets": len(all_seeds), "groups": records, "sem_equivalence": equivalence,
        "generator_sha256": file_hash(Path(__file__))}


def main(argv=None):
    args = parse_args(argv)
    catalogue = load_catalogue(args.sem_file)
    equivalence = check_sem_equivalence(catalogue)
    config = run_config(args, catalogue)
    root = args.output_root
    if args.validate_only and not root.is_dir():
        raise FileNotFoundError(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".generation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another generator is using this output root") from None
        config_path = root / "generation_config.json"
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise ValueError("Existing generation protocol differs; use a NEW output root. Nothing overwritten.")
            if json.loads((root / "catalogue.json").read_text()) != catalogue_document(catalogue):
                raise ValueError("Saved SEM catalogue changed")
            if file_hash(root / "all_25_sems.tex") != catalogue.source_sha256:
                raise ValueError("Saved SEM specification changed")
        else:
            if args.validate_only or any(p.name != ".generation.lock" for p in root.iterdir()):
                raise ValueError("Not an empty/matching posterior-data directory; refusing to overwrite")
            atomic_json(root / "catalogue.json", catalogue_document(catalogue))
            atomic_bytes(root / "all_25_sems.tex", catalogue.source.read_bytes())
            atomic_json(config_path, config)
        jobs = jobs_for(args)
        print(f"SEM CHECK PASSED: 25 DAGs / 11 MECs; shared-noise error={equivalence['max_shared_noise_sem_matrix_error']:.3g}", flush=True)
        print(f"datasets={sum(job['count'] for job in jobs)} shards={len(jobs)} method={args.sampling_method}; no normalization", flush=True)
        if not args.validate_only:
            for job in jobs:
                write_shard(config, job, catalogue, args.batch_size)
        report = final_report(args, config, catalogue, jobs, equivalence)
        atomic_json(root / "validation_report.json", report)
        print(f"POSTERIOR DATA COMPLETE: datasets={report['total_datasets']} output_root={root}", flush=True)


if __name__ == "__main__":
    main()
