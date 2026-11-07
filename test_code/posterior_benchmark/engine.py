"""Shared training/evaluation engine; only raw X and sampled G enter the loss."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import tempfile
import time

import numpy as np
import torch

from test_code.posterior_benchmark import generate_training_data as gen
from test_code.posterior_benchmark.data import (
    batch_plan, manifest_fingerprint, read_batch, sample_training_observations,
)
from test_code.posterior_benchmark.metrics import distribution_metrics, summarize, write_csv
from test_code.posterior_benchmark.models import ModelConfig, PosteriorModel


def torch_load(path, device="cpu"):
    # These are this experiment's own checkpoints, never arbitrary downloaded pickles.
    return torch.load(path, map_location=device, weights_only=False)


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, name)
        with open(name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(name, destination)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def device_for(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def run_key(method, seed):
    return f"{method}_seed{seed}"


@contextmanager
def finish_batch_on_signal():
    stop = {"requested": False}
    previous = {}
    def handler(signum, frame):
        stop["requested"] = True
        print("Stop requested; finishing this batch and saving a resumable checkpoint.", flush=True)
    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.signal(number, handler)
    try:
        yield stop
    finally:
        for number, handler_before in previous.items():
            signal.signal(number, handler_before)


def train_chunk(run_root, protocol, protocol_hash, catalogue, method, seed,
                chunk_index, manifest, device):
    config = ModelConfig(**{**protocol["model"], "reverse_steps": tuple(protocol["model"]["reverse_steps"])})
    train = protocol["training"]
    key = run_key(method, seed)
    directory = Path(run_root) / "checkpoints" / key
    latest, archive = directory / "latest.pt", directory / f"checkpoint_chunk_{chunk_index:05d}.pt"
    fingerprint = manifest_fingerprint(manifest)
    batches = batch_plan(manifest, train["batch_size"], seed=protocol["data_seed"] + chunk_index)
    count = sum(record["count"] for record in manifest)
    set_seed(seed)
    np.random.seed(protocol["data_seed"] % (2**32))
    model = PosteriorModel(method, config, catalogue.graphs).to(device)
    parameter_counts = {"total": sum(p.numel() for p in model.parameters()),
                        "encoder": sum(p.numel() for p in model.encoder.parameters())}
    optimizer = torch.optim.AdamW(model.parameters(), lr=train["learning_rate"],
                                 weight_decay=train["weight_decay"])
    warmup = train["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1., (step + 1) / max(1, warmup)))
    step = seen = cursor = 0
    loss_sum = 0.
    chunk_seen = 0
    history = []
    sampling_history = []
    observations_seen = 0
    component_sums = {}
    sampling_trace = hashlib.sha256(f"{protocol_hash}/chunk-{chunk_index}".encode()).hexdigest()
    n_histogram = {}
    if latest.exists():
        state = torch_load(latest)
        if (state["protocol_hash"] != protocol_hash or state["method"] != method
            or state["training_seed"] != seed):
            raise ValueError(f"Checkpoint protocol mismatch: {latest}")
        if state["completed_chunks"] >= chunk_index:
            if not archive.is_file():
                if state["completed_chunks"] == chunk_index:
                    # Recovery from a stop between the two atomic file commits.
                    atomic_copy(latest, archive)
                else:
                    raise ValueError(f"Missing completed checkpoint: {archive}")
            if state["data_history"][chunk_index - 1] != fingerprint:
                raise ValueError("Completed chunk observations changed")
            print(f"[{key}] chunk={chunk_index} already trained; skip", flush=True)
            return archive, True
        if state["completed_chunks"] != chunk_index - 1:
            raise ValueError("Cannot skip an untrained chunk")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        # map_location=cpu is deliberate; migrate optimizer tensors explicitly.
        for values in optimizer.state.values():
            for name, tensor in values.items():
                if torch.is_tensor(tensor):
                    # Non-capturable AdamW keeps its scalar step on the CPU.
                    values[name] = tensor.cpu() if name == "step" else tensor.to(device)
        scheduler.load_state_dict(state["scheduler"])
        step, seen, history = state["global_step"], state["datasets_seen"], state["data_history"]
        sampling_history = state["sampling_history"]
        observations_seen = state["observations_seen"]
        if state["active_chunk"] == chunk_index:
            if state["active_data_fingerprint"] != fingerprint:
                raise ValueError("Active training observations changed during resume")
            cursor, loss_sum, chunk_seen = state["next_batch"], state["loss_sum"], state["chunk_seen"]
            component_sums = state["component_sums"]
            sampling_trace, n_histogram = state["sampling_trace_sha256"], state["minibatch_n_histogram"]
        restore_rng(state["rng"])
    elif chunk_index != 1:
        raise ValueError(f"Missing preceding checkpoint for {key}")
    expected_seen = count * (chunk_index - 1) + chunk_seen
    if seen != expected_seen or cursor > len(batches):
        raise ValueError("Dataset counter / batch cursor mismatch")

    def save(completed=False):
        state = {"format": "posterior-d3-independent-training-v3", "protocol": protocol,
                 "protocol_hash": protocol_hash, "method": method, "training_seed": seed,
                 "parameter_counts": parameter_counts,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "rng": rng_state(),
                 "global_step": step, "datasets_seen": seen, "active_chunk": chunk_index,
                 "observations_seen": observations_seen,
                 "active_data_fingerprint": fingerprint, "next_batch": cursor,
                 "completed_chunks": chunk_index if completed else chunk_index - 1,
                 "data_history": history + ([fingerprint] if completed else []),
                 "sampling_history": sampling_history + ([sampling_trace] if completed else []),
                 "sampling_trace_sha256": sampling_trace, "minibatch_n_histogram": n_histogram,
                 "loss_sum": loss_sum, "chunk_seen": chunk_seen, "component_sums": component_sums}
        atomic_torch_save(latest, state)
        if completed:
            atomic_copy(latest, archive)

    model.train()
    start_time = time.monotonic()
    with finish_batch_on_signal() as stop:
        while cursor < len(batches):
            record, indices = batches[cursor]
            # No posterior field is read by the training path.
            batch = read_batch(record, indices, posterior=False)
            observation_indices = sample_training_observations(batch, protocol["observation_sampling"])
            data = torch.as_tensor(batch["data"][:, observation_indices, :], device=device, dtype=torch.float32)
            classes = torch.as_tensor(batch["class_index"].astype(np.int64), device=device)
            optimizer.zero_grad(set_to_none=True)
            components = model.training_loss(
                data, classes, diffusion_loss=train["diffusion_loss"],
                order_loss_weight=train["order_loss_weight"],
                skeleton_loss_weight=train["skeleton_loss_weight"])
            loss = components["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss: {key}, chunk={chunk_index}, batch={cursor}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train["gradient_clip"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient; training data retained")
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(indices)
            for name, value in components.items():
                if name != "loss":
                    component_sums[name] = component_sums.get(name, 0.) + float(value.detach()) * len(indices)
            chunk_seen += len(indices)
            seen += len(indices)
            observations_seen += len(indices) * len(observation_indices)
            trace = hashlib.sha256(bytes.fromhex(sampling_trace))
            trace.update(np.asarray(batch["dataset_seed"], dtype="<u8").tobytes())
            trace.update(np.asarray(observation_indices, dtype="<i8").tobytes())
            sampling_trace = trace.hexdigest()
            n_histogram[str(len(observation_indices))] = n_histogram.get(str(len(observation_indices)), 0) + 1
            step += 1
            cursor += 1
            if cursor % train["progress_every"] == 0 or cursor == len(batches):
                print(f"[{key}] chunk={chunk_index} batch={cursor}/{len(batches)} "
                      f"datasets_seen={seen} n={len(observation_indices)} "
                      f"loss={loss_sum/chunk_seen:.6f} "
                      + " ".join(f"{name}={value/chunk_seen:.6f}" for name, value in component_sums.items()) + " "
                      f"lr={scheduler.get_last_lr()[0]:.3g} elapsed={time.monotonic()-start_time:.1f}s", flush=True)
            if cursor % train["checkpoint_every"] == 0 or stop["requested"]:
                save()
            if stop["requested"]:
                return latest, False
        if chunk_seen != count:
            raise ValueError("Did not consume each training dataset exactly once")
        save(completed=True)
    return archive, True


def evaluate_checkpoint(checkpoint, manifest, catalogue, output_dir, device,
                        batch_size=16, save_predictions=True, overwrite=False):
    output_dir = Path(output_dir)
    checkpoint = Path(checkpoint)
    state = torch_load(checkpoint)
    if state["protocol"]["sem_sha256"] != catalogue.source_sha256:
        raise ValueError("Checkpoint and evaluation SEM prior differ")
    identity = {"checkpoint_sha256": gen.file_hash(checkpoint),
                "data_fingerprint": manifest_fingerprint(manifest),
                "method": state["method"], "training_seed": state["training_seed"],
                "datasets_seen": state["datasets_seen"], "save_predictions": save_predictions,
                "metric_version": "exact-graph-posterior-v1"}
    done = output_dir / "evaluation_complete.json"
    if done.exists() and not overwrite:
        old = json.loads(done.read_text())
        if old["identity"] != identity:
            raise ValueError(f"Different checkpoint/data already scored at {output_dir}; use a new output or --overwrite")
        for relative, digest in old["files"].items():
            if gen.file_hash(output_dir / relative) != digest:
                raise ValueError(f"Incomplete/changed evaluation file: {relative}")
        print(f"[EVALUATION VERIFIED/SKIP] {output_dir}", flush=True)
        return old["summary"]
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ModelConfig(**{**state["protocol"]["model"],
                            "reverse_steps": tuple(state["protocol"]["model"]["reverse_steps"])})
    model = PosteriorModel(state["method"], config, catalogue.graphs).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    rows, predictions = [], {}
    plan = batch_plan(manifest, batch_size)
    with torch.inference_mode():
        for batch_index, (record, indices) in enumerate(plan, 1):
            batch = read_batch(record, indices, posterior=True)
            x = torch.as_tensor(batch["data"], device=device, dtype=torch.float32)
            pred_log = model(x).cpu().numpy()
            metrics = distribution_metrics(batch["log_posterior"], pred_log,
                                           batch["class_index"].astype(int), catalogue.mec_ids)
            for index in range(len(indices)):
                row = {"method": state["method"], "training_seed": state["training_seed"],
                       "sample_size": record["sample_size"],
                       "dataset_seed": int(batch["dataset_seed"][index]),
                       "class_index": int(batch["class_index"][index]),
                       "mec_id": int(batch["mec_id"][index])}
                row.update({name: float(values[index]) for name, values in metrics.items()})
                rows.append(row)
            if save_predictions:
                group = predictions.setdefault(record["sample_size"], {})
                for name, value in {"pred_log_posterior": pred_log,
                                    "posterior": batch["posterior"],
                                    "log_posterior": batch["log_posterior"],
                                    "dataset_seed": batch["dataset_seed"],
                                    "data_sha256": batch["data_sha256"],
                                    "class_index": batch["class_index"], "mec_id": batch["mec_id"]}.items():
                    group.setdefault(name, []).append(value)
            if batch_index % 100 == 0 or batch_index == len(plan):
                print(f"[evaluate {state['method']}/seed{state['training_seed']}] "
                      f"batches={batch_index}/{len(plan)} rows={len(rows)}", flush=True)
    summary = summarize(rows)
    write_csv(output_dir / "raw_metrics.csv", rows)
    write_csv(output_dir / "summary_by_samples.csv", summary)
    paths = ["raw_metrics.csv", "summary_by_samples.csv"]
    for n, fields in predictions.items():
        payload = {name: np.concatenate(parts) for name, parts in fields.items()}
        # HDF5 fixed-string dtypes carry h5py-only metadata; the byte strings
        # themselves are sufficient and safe in a portable allow_pickle=False NPZ.
        payload = {name: value.view(np.dtype(value.dtype.str)) if value.dtype.metadata else value
                   for name, value in payload.items()}
        payload["pred_posterior"] = np.exp(payload["pred_log_posterior"])
        payload["mec_ids"] = catalogue.mec_ids
        payload["graph_ids"] = np.arange(1, 26)
        payload["sample_size"] = np.array(n)
        relative = f"predictions/n{n}.npz"
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".n{n}.", suffix=".tmp", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(handle, **payload)
            os.replace(name, destination)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        paths.append(relative)
    report = {"identity": identity, "summary": summary,
              "files": {name: gen.file_hash(output_dir / name) for name in paths}}
    gen.atomic_json(done, report)
    return summary
