"""Reuse the checked SEM generator; feed unchanged raw observations to models."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from test_code.posterior_benchmark import generate_training_data as gen


@contextmanager
def exclusive_lock(path):
    with Path(path).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process holds {path}") from None
        yield


def derived_seed(master, namespace):
    digest = hashlib.sha256(f"posterior-stream-v1/{master}/{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def prepare_splits(root, catalogue, sample_sizes, counts, splits, seed,
                   datasets_per_file=5000, batch_size=128):
    """Create/verify only the requested splits, with the original generator."""
    root = Path(root).resolve()
    arguments = ["--output-root", str(root), "--sem-file", str(catalogue.source),
                 "--sample-sizes", ",".join(map(str, sample_sizes)),
                 "--datasets-per-file", str(datasets_per_file),
                 "--batch-size", str(batch_size), "--seed", str(seed)]
    for split in gen.SPLITS:
        arguments.extend([f"--{split}-per-size", str(counts[split])])
    args = gen.parse_args(arguments)
    config = gen.run_config(args, catalogue)
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / ".generation.lock"):
        path = root / "generation_config.json"
        if path.exists():
            if json.loads(path.read_text()) != config:
                raise ValueError(f"Generation protocol changed: {root}; use a new run")
            if json.loads((root / "catalogue.json").read_text()) != gen.catalogue_document(catalogue):
                raise ValueError("Saved catalogue changed")
            if gen.file_hash(root / "all_25_sems.tex") != catalogue.source_sha256:
                raise ValueError("Saved SEM document changed")
        else:
            if any(p.name != ".generation.lock" for p in root.iterdir()):
                raise ValueError(f"Refusing to write into an unrelated directory: {root}")
            gen.atomic_json(root / "catalogue.json", gen.catalogue_document(catalogue))
            gen.atomic_bytes(root / "all_25_sems.tex", catalogue.source.read_bytes())
            gen.atomic_json(path, config)
        for job in gen.jobs_for(args):
            if job["split"] in splits:
                gen.write_shard(config, job, catalogue, batch_size)
        manifest = {split: split_manifest(root, split) for split in splits}
        gen.atomic_json(root / "prepared_splits.json", {
            split: {"datasets": sum(row["count"] for row in rows),
                    "fingerprint": manifest_fingerprint(rows)} for split, rows in manifest.items()
        })
    return manifest


def split_manifest(root, split, verify=False):
    root = Path(root).resolve()
    if split not in gen.SPLITS:
        raise ValueError(split)
    config = json.loads((root / "generation_config.json").read_text())
    catalogue = gen.load_catalogue(root / "all_25_sems.tex") if verify else None
    result = []
    expected = int(config["counts_per_size"][split])
    for n in config["sample_sizes"]:
        count = 0
        paths = sorted((root / split / f"n{n}").glob("d3_shard_*.hdf5"))
        shard_size = int(config["datasets_per_file"])
        expected_paths = [root / split / f"n{n}" / f"d3_shard_{i:05d}.hdf5"
                          for i, start in enumerate(range(0, expected, shard_size))]
        if paths != expected_paths:
            raise ValueError(f"Missing/extra shards in {root}/{split}/n{n}")
        for shard, path in enumerate(paths):
            with h5py.File(path, "r") as h:
                length = len(h["data"])
                if (h.attrs.get("generation_config") != gen.canonical(config)
                    or h.attrs.get("split") != split or int(h.attrs["n"]) != n
                    or h["data"].shape != (length, n, 3)
                    or length != min(shard_size, expected - shard * shard_size)
                    or int(h.attrs["start"]) != shard * shard_size
                    or int(h.attrs["shard"]) != shard
                    or not h["completed"][:].all()):
                    raise ValueError(f"Incomplete or mismatched shard: {path}")
                if verify:
                    job = {"split": split, "n": n, "shard": shard,
                           "start": shard * shard_size, "count": length}
                    gen.validate_handle(h, config, job, catalogue, batch_size=128)
                digest = hashlib.sha256(gen.canonical(config).encode())
                for field in ("dataset_seed", "class_index", "data_sha256"):
                    digest.update(h[field][:].tobytes())
                result.append({"path": str(path), "sample_size": n,
                               "count": length, "fingerprint": digest.hexdigest()})
                count += length
        if count != expected:
            raise ValueError(f"Missing rows in {root}/{split}/n{n}: {count}/{expected}")
    return result


def manifest_fingerprint(manifest):
    # Relative logical content, not the server/local absolute path.
    rows = [(r["sample_size"], Path(r["path"]).name, r["count"], r["fingerprint"])
            for r in manifest]
    return hashlib.sha256(gen.canonical(rows).encode()).hexdigest()


def batch_plan(manifest, batch_size, seed=None):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rng = np.random.default_rng(seed) if seed is not None else None
    batches = []
    for record in manifest:
        indices = np.arange(record["count"])
        if rng is not None:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batches.append((record, indices[start:start + batch_size].copy()))
    if rng is not None:
        rng.shuffle(batches)
    return batches


def sample_training_observations(batch, sampling):
    """The original per-minibatch sampler, using this run's own NumPy RNG."""
    from ml2_meta_causal_discovery.utils.datautils import shared_batch_observation_indices
    return shared_batch_observation_indices([(x, None) for x in batch["data"]], **sampling)


def prepare_fixed_evaluation_data(run_root, protocol, catalogue, split):
    """Called manually by evaluation, NEVER by the training launcher."""
    root = Path(run_root) / "data" / "fixed"
    counts = {"train": 25,  # unused placeholder, independent of training budget/method
              "val": protocol["evaluation_counts_per_size"]["val"],
              "test": protocol["evaluation_counts_per_size"]["test"]}
    return prepare_splits(root, catalogue, protocol["sample_sizes"], counts, (split,),
                          derived_seed(protocol["evaluation_seed"], "fixed-validation-test"),
                          protocol["evaluation_datasets_per_file"], batch_size=128)[split]


def read_batch(record, indices, posterior=False, observation_indices=None):
    """Read fancy-indexed HDF5 rows in requested order; never normalize/sample X."""
    indices = np.asarray(indices)
    sorting = np.argsort(indices)
    inverse = np.argsort(sorting)
    with h5py.File(record["path"], "r") as h:
        names = ["data", "class_index", "dataset_seed", "data_sha256"]
        if posterior:
            names += ["posterior", "log_posterior", "mec_id"]
        batch = {name: h[name][indices[sorting]][inverse] for name in names}
    if observation_indices is not None:
        if posterior:
            raise ValueError("Stored full-data posterior cannot label a subsample; recompute its posterior")
        batch["data"] = batch["data"][:, observation_indices, :]
    return batch
