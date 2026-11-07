#!/usr/bin/env python3
"""Generate independent observational Station1 replicates in unified HDF5 format.

Fit the official causalAssembly DRFs ONCE under the six-node Station1 DAG.
The generator is cached locally; observations are newly sampled, not selected
from the source table. All replicates share the same DAG and fitted mechanisms.
No intervention API, extra standardization, or causal-discovery model is used.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
import time

import h5py
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import pandas as pd

SAMPLE_SIZES = (20, 50, 100, 300, 1000, 2000)
DISTRIBUTION = "causalassembly_station1"
METADATA_DTYPE = np.dtype([
    ("dataset_seed", "<u8"), ("replicate", "<u4"),
    ("sem_type", "u1"), ("graph_type", "u1"),
    ("noise_mode", "i1"), ("activation", "i1"),
])


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sample_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f4").tobytes()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def parse_sample_sizes(value):
    try:
        result = tuple(sorted(int(part) for part in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integers") from exc
    if not result or min(result) < 2 or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("Sample sizes must be unique integers >= 2")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Official causalAssembly repository, with data/ and causalAssembly/.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-sizes", type=parse_sample_sizes, default=SAMPLE_SIZES)
    parser.add_argument("--datasets-per-cell", type=int, default=100)
    parser.add_argument("--fit-rows", type=int, default=0, help="Reference rows for ONE DRF fit; 0 = all released rows. 500 matches the README demonstration, not the full-data protocol.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-batch-rows", type=int, default=2048, help="Limit DRF conditional-weight matrix memory. Keep unchanged when resuming.")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.datasets_per_cell < 1 or args.fit_rows < 0 or args.sampling_batch_rows < 1 or not 0 <= args.seed < 2**31:
        parser.error("Invalid dataset count, fit-row count, sampling batch size or seed")
    args.source_root = args.source_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    return args


def load_reference(source_root, fit_rows, seed):
    data_path = source_root / "data/data_sets/synthetic_data.csv"
    graph_path = source_root / "data/ground_truth/ground_truth.json"
    data = pd.read_csv(data_path)
    graph = json_graph.adjacency_graph(json.loads(graph_path.read_text()))
    if not nx.is_directed_acyclic_graph(graph) or set(data.columns) != set(graph.nodes):
        raise ValueError("Reference DAG/data mismatch")
    nodes = sorted([node for node in graph.nodes if node.startswith("Station1_")], key=lambda node: int(node.rsplit("_", 1)[1]))
    subgraph = nx.DiGraph()
    subgraph.add_nodes_from(nodes)
    subgraph.add_edges_from(graph.subgraph(nodes).edges)
    if len(nodes) != 6 or subgraph.number_of_edges() != 7:
        raise ValueError("Expected the official Station1 reference DAG: 6 nodes, 7 edges")
    frame = data.loc[:, nodes]
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Non-finite reference observations")
    if fit_rows:
        if not 30 <= fit_rows <= len(frame):
            raise ValueError(f"Use fit-rows=0 or a value in [30,{len(frame)}]")
        frame = frame.sample(n=fit_rows, random_state=seed, replace=False)
    return frame, subgraph, nodes, {
        "data_sha256": sha256(data_path), "graph_sha256": sha256(graph_path),
        "source_rows": len(data), "actual_fit_rows": len(frame),
        "fit_row_indices_sha256": hashlib.sha256(np.asarray(frame.index, dtype="<i8").tobytes()).hexdigest(),
    }


def generation_protocol(args, frame, graph, columns, source):
    # Import R only after verifying the installation. Never let the library
    # trigger its interactive auto-install of DRF during a generation run.
    import rpy2.robjects as ro
    if not bool(ro.r["requireNamespace"]("drf", quietly=True)[0]):
        raise RuntimeError("Install the R drf package before running this script")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.source_root, text=True).strip()
    return {
        "format": "causalassembly_station1_independent_replicates_v1",
        "dataset_type": "semisynthetic_observational", "distribution": DISTRIBUTION,
        "columns": columns, "node_count": 6, "edges": 7,
        "directed_edges": sorted([list(edge) for edge in graph.edges]),
        "label_convention": "label[i, j] = 1 means i -> j",
        "source": source, "official_source_commit": commit,
        "source_code_sha256": {name: sha256(args.source_root / "causalAssembly" / name) for name in ("models_dag.py", "drf_fitting.py")},
        "versions": {name: version(name) for name in ("causalAssembly", "numpy", "pandas", "networkx", "rpy2")},
        "r_version": str(ro.r('paste(R.version$major, R.version$minor, sep=".")')[0]),
        "drf_version": str(ro.r('as.character(packageVersion("drf"))')[0]),
        "fit_seed": args.seed, "sampling_seed": args.seed,
        "drf": {"fit_api": "causalAssembly.drf_fitting.fit_drf", "num_trees": 2000, "min_node_size": 15, "splitting_rule": "FourierMMD", "fit_count": 1},
        "sample_sizes": list(args.sample_sizes), "datasets_per_cell": args.datasets_per_cell,
        "sampling_batch_rows": args.sampling_batch_rows,
        "sampling": "Independent SeedSequence([seed,n,replicate,1]) streams; official sample_from_drf in bounded row blocks; not nested subsets",
        "smoothed_root_sampling": True, "interventions": False,
        "extra_standardization": False, "storage_dtype": "float32",
        "all_replicates_share_graph_and_fitted_mechanisms": True,
    }


def dataset_seed(seed, sample_size, replicate):
    return int(np.random.SeedSequence([seed, sample_size, replicate, 1]).generate_state(1, dtype=np.uint64)[0])


def expected_metadata(protocol, n):
    count = protocol["datasets_per_cell"]
    result = np.zeros(count, dtype=METADATA_DTYPE)
    result["dataset_seed"] = [dataset_seed(protocol["sampling_seed"], n, i) for i in range(count)]
    result["replicate"] = np.arange(count, dtype=np.uint32)
    result["activation"] = -1
    return result


def shard_path(output_root, n):
    return output_root / DISTRIBUTION / f"d6/n{n}/test/d6_shard_00000.hdf5"


def partial_path(path):
    # Do not let the unified manifest discover an unfinished shard.
    return path.with_name(f".{path.stem}.partial.hdf5")


def check_header(handle, protocol, n, dag):
    count = protocol["datasets_per_cell"]
    if handle.attrs.get("generation_protocol_sha256") != canonical_hash(protocol):
        raise ValueError("HDF5 generation protocol mismatch; choose a new output directory")
    if handle["data"].shape != (count, n, 6) or handle["data"].dtype != np.dtype("float32"):
        raise ValueError("Wrong HDF5 data shape or dtype")
    if handle["label"].shape != (count, 6, 6) or not np.array_equal(handle["label"][:], np.broadcast_to(dag, (count, 6, 6))):
        raise ValueError("Wrong ground-truth label shape or orientation")
    if not np.array_equal(handle["metadata"][:], expected_metadata(protocol, n)):
        raise ValueError("Wrong per-dataset seeds/metadata")
    if json.loads(str(handle.attrs["columns_json"])) != protocol["columns"]:
        raise ValueError("HDF5 column order differs from reference graph")
    if handle["completed"].shape != (count,) or handle["sample_sha256"].shape != (count,):
        raise ValueError("Wrong completion/hash arrays")


def check_completed_rows(handle):
    completed = np.asarray(handle["completed"][:], dtype=bool)
    for index in np.flatnonzero(completed):
        array = handle["data"][index]
        if not np.isfinite(array).all() or sample_hash(array) != handle["sample_sha256"][index].decode():
            raise ValueError(f"Corrupt/non-finite completed dataset at row {index}")
    return completed


def validate_shard(path, protocol, n, dag):
    with h5py.File(path, "r") as handle:
        check_header(handle, protocol, n, dag)
        completed = check_completed_rows(handle)
        if not completed.all():
            raise ValueError(f"Incomplete final shard: {path}")
    return {"sample_size": n, "datasets": len(completed), "shape": [len(completed), n, 6], "finite": True}


def create_partial(path, protocol, n, dag):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = protocol["datasets_per_cell"]
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        with h5py.File(name, "w") as handle:
            handle.create_dataset("data", shape=(count, n, 6), dtype="f4", chunks=(1, min(n, 256), 6), compression="lzf")
            handle.create_dataset("label", data=np.broadcast_to(dag, (count, 6, 6)), compression="lzf")
            handle.create_dataset("metadata", data=expected_metadata(protocol, n))
            handle.create_dataset("completed", data=np.zeros(count, dtype=bool))
            handle.create_dataset("sample_sha256", shape=(count,), dtype="S64")
            handle.attrs.update(distribution=DISTRIBUTION, n_nodes=6, num_samples=n,
                                num_datasets=count, edges=7, dataset_type="semisynthetic_observational",
                                label_convention=protocol["label_convention"],
                                generation_protocol_sha256=canonical_hash(protocol),
                                columns_json=json.dumps(protocol["columns"]),
                                sem_type_codes=json.dumps({0: "causalassembly_drf"}),
                                graph_type_codes=json.dumps({0: "station1_fixed_dag"}),
                                noise_mode_codes=json.dumps({0: "observational_drf"}),
                                activation_codes=json.dumps({-1: "not_applicable"}))
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def generate_cell(path, protocol, n, dag, cell):
    if path.exists():
        validate_shard(path, protocol, n, dag)
        print(f"n={n}: verified existing {protocol['datasets_per_cell']} datasets; skipped", flush=True)
        return
    partial = partial_path(path)
    if not partial.exists():
        create_partial(partial, protocol, n, dag)
    with h5py.File(partial, "r+") as handle:
        check_header(handle, protocol, n, dag)
        completed = check_completed_rows(handle)
        count = len(completed)
        for index in range(count):
            if completed[index]:
                continue
            started = time.monotonic()
            cell.random_state = np.random.default_rng(int(handle["metadata"][index]["dataset_seed"]))
            for start in range(0, n, protocol["sampling_batch_rows"]):
                stop = min(start + protocol["sampling_batch_rows"], n)
                frame = cell.sample_from_drf(size=stop - start)
                if list(frame.columns) != protocol["columns"] or frame.shape != (stop - start, 6):
                    raise ValueError("Official sampler returned wrong shape or column order")
                array = frame.to_numpy(dtype=np.float32)
                if not np.isfinite(array).all():
                    raise ValueError("Official sampler returned non-finite observations")
                handle["data"][index, start:stop] = array
            handle["sample_sha256"][index] = sample_hash(handle["data"][index]).encode()
            handle.flush()
            handle["completed"][index] = True
            handle.flush()
            print(f"n={n} [{index + 1}/{count}] saved; seconds={time.monotonic() - started:.2f}", flush=True)
    validate_shard(partial, protocol, n, dag)
    os.replace(partial, path)
    print(f"CELL COMPLETE: {path}", flush=True)


def load_or_fit(args, frame, graph, protocol):
    import rpy2.robjects as ro
    import causalAssembly.models_dag as models_dag
    from causalAssembly.drf_fitting import fit_drf
    from causalAssembly.models_dag import ProcessCell
    if Path(models_dag.__file__).resolve() != args.source_root / "causalAssembly/models_dag.py":
        raise RuntimeError("Installed causalAssembly is not the --source-root checkout; run pip install -e . there first")
    cache_dir = args.output_root / "generator"
    cache = cache_dir / "station1_drf.pkl"
    manifest = cache_dir / "cache_manifest.json"
    if cache.is_file() and manifest.is_file():
        record = json.loads(manifest.read_text())
        if record["protocol_sha256"] != canonical_hash(protocol) or record["pickle_sha256"] != sha256(cache):
            raise ValueError("Generator cache changed or has a different protocol")
        # Only load this script's own trusted local cache, never downloaded pickles.
        with cache.open("rb") as handle:
            cell = pickle.load(handle)
        if cell.nodes != protocol["columns"] or cell.num_edges != 7:
            raise ValueError("Cached generator has the wrong graph")
        print(f"Loaded fitted generator: {cache}", flush=True)
        return cell
    if cache.exists() or manifest.exists():
        raise RuntimeError("Incomplete generator cache; keep it for inspection and use a new output directory")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cell = ProcessCell(name="Station1")
    cell.graph = graph.copy()
    ro.r["set.seed"](args.seed)
    print(f"Fitting ONCE: reference_rows={len(frame)}, nodes=6, edges=7, official num_trees=2000; CPU step may take time", flush=True)
    started = time.monotonic()
    cell.drf = fit_drf(cell, data=frame)
    descriptor, name = tempfile.mkstemp(prefix=".station1_drf.", suffix=".tmp", dir=cache_dir)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(cell, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(name, cache)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    atomic_json(manifest, {"protocol_sha256": canonical_hash(protocol), "pickle_sha256": sha256(cache), "fit_seconds": time.monotonic() - started})
    print(f"GENERATOR FIT COMPLETE: {cache}", flush=True)
    return cell


def validate_benchmark(output_root, protocol, dag):
    records = [validate_shard(shard_path(output_root, n), protocol, n, dag) for n in protocol["sample_sizes"]]
    return {"status": "COMPLETE", "distribution": DISTRIBUTION, "node_count": 6, "edges": 7,
            "cells": records, "total_datasets": sum(record["datasets"] for record in records),
            "protocol_sha256": canonical_hash(protocol), "all_replicates_share_graph_and_fitted_mechanisms": True}


def main(argv=None):
    args = parse_args(argv)
    frame, graph, columns, source = load_reference(args.source_root, args.fit_rows, args.seed)
    protocol = generation_protocol(args, frame, graph, columns, source)
    dag = nx.to_numpy_array(graph, nodelist=columns, dtype=np.int8, weight=None)
    config = args.output_root / "generation_config.json"
    if args.validate_only:
        if json.loads(config.read_text()) != protocol:
            raise ValueError("Requested protocol differs from the stored generation configuration")
        report = validate_benchmark(args.output_root, protocol, dag)
        print(json.dumps(report, indent=2), flush=True)
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / ".generation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another generator is using this output directory") from None
        if config.exists():
            if json.loads(config.read_text()) != protocol:
                raise ValueError("Existing output has a different protocol; choose a NEW directory. No files overwritten.")
        else:
            if any(path.name != ".generation.lock" for path in args.output_root.iterdir()):
                raise FileExistsError("Non-empty output directory without generation_config.json")
            atomic_json(config, protocol)
        remaining = []
        for n in args.sample_sizes:
            path = shard_path(args.output_root, n)
            if path.exists():
                validate_shard(path, protocol, n, dag)
            else:
                remaining.append(n)
        if remaining:
            cell = load_or_fit(args, frame, graph, protocol)
            for n in remaining:
                generate_cell(shard_path(args.output_root, n), protocol, n, dag, cell)
        report = validate_benchmark(args.output_root, protocol, dag)
        atomic_json(args.output_root / "validation_report.json", report)
        print(f"STATION1 BENCHMARK COMPLETE: datasets={report['total_datasets']} cells={len(report['cells'])} nodes=6 edges=7\noutput_root={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
