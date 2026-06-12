"""
Train a transformer neural process on the causal classification task.
"""
import argparse
import json
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import wandb

from ml2_meta_causal_discovery.models.causaltransformernp import (
    AviciDecoder,
    CausalProbabilisticDecoder,
    CsivaDecoder,
)
from ml2_meta_causal_discovery.models.topo_order_diffusion import (
    CausalTopoOrderDiffusion,
    CausalPriorityTopoOrderDiffusion,
)
try:
    from ml2_meta_causal_discovery.models.causaltransformernp import CausalProbabilisticARDecoder
except ImportError:
    CausalProbabilisticARDecoder = None
from ml2_meta_causal_discovery.utils.args import retun_default_args
from ml2_meta_causal_discovery.utils.datautils import (
    MultipleFileDataset, MultipleFileDatasetWithPadding)
from ml2_meta_causal_discovery.utils.train_classifier_model import \
    CausalClassifierTrainer


def optional_root(path_arg):
    return Path(path_arg).expanduser().resolve() if path_arg else None


def resolve_synth_data_root(args, work_dir: Path):
    root = optional_root(args.synth_data_root)
    if root is not None:
        return root
    return work_dir / "datasets" / "data" / "synth_training_data"


def resolve_models_root(args, work_dir: Path):
    root = optional_root(args.models_root)
    if root is not None:
        return root
    return work_dir / "experiments" / "causal_classification" / "models"


def resolve_results_root(args, work_dir: Path):
    root = optional_root(args.results_root)
    if root is not None:
        return root
    return work_dir / "experiments" / "causal_classification" / "results"


def resolve_init_checkpoint(args, models_root: Path):
    if args.init_from_path:
        return Path(args.init_from_path).expanduser().resolve()

    if args.init_from_run_name or args.init_from_checkpoint:
        if not args.init_from_run_name or not args.init_from_checkpoint:
            raise ValueError(
                "Use both --init_from_run_name and --init_from_checkpoint, "
                "or pass --init_from_path directly."
            )
        return models_root / args.init_from_run_name / args.init_from_checkpoint

    return None


def resolve_resume_checkpoint(args, models_root: Path):
    if args.resume_from_path:
        return Path(args.resume_from_path).expanduser().resolve()

    if args.resume_from_run_name or args.resume_from_checkpoint:
        if not args.resume_from_run_name or not args.resume_from_checkpoint:
            raise ValueError(
                "Use both --resume_from_run_name and --resume_from_checkpoint, "
                "or pass --resume_from_path directly."
            )
        return models_root / args.resume_from_run_name / args.resume_from_checkpoint

    return None


def torch_load(path: Path, device: str, weights_only: bool = False):
    try:
        return torch.load(path, map_location=device, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=device)


def load_initial_weights(model, checkpoint_path: Path, device: str):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}")
    state_dict = torch_load(checkpoint_path, device=device, weights_only=True)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    print(f"Initialized model weights from: {checkpoint_path}")


def move_optimizer_state_to_param_device_and_dtype(optimizer):
    for param, state in optimizer.state.items():
        if not isinstance(state, dict):
            continue
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            if value.is_floating_point() and value.ndim > 0 and param.is_floating_point():
                state[key] = value.to(device=param.device, dtype=param.dtype)
            else:
                state[key] = value.to(device=param.device)


def load_training_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint_path: Path,
    device: str,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    checkpoint = torch_load(checkpoint_path, device=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Resume checkpoint must be a full training checkpoint with "
            "'model_state_dict', not a raw model state_dict."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state_to_param_device_and_dtype(optimizer)

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    elif scheduler_state is not None:
        print("Resume checkpoint has scheduler state, but no scheduler is enabled.")

    start_epoch = int(checkpoint.get("epoch", 0))
    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Resumed full training state from: {checkpoint_path} "
        f"(start_epoch={start_epoch}, lr={current_lr})"
    )
    return start_epoch


def evaluate_topo_order_model(trainer, num_samples: int = 1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if getattr(trainer, "bfloat16", False) else torch.float32
    model = trainer.model.to(device=device, dtype=dtype)
    model.eval()
    all_edge_acc = []
    all_topo_valid = []

    with torch.no_grad():
        for data in trainer.test_loader:
            inputs, targets, attention_mask = data
            inputs = inputs.to(device, dtype=dtype)
            targets = targets.to(device, dtype=torch.float32)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, dtype=dtype)
            inputs = (inputs - inputs.mean(dim=1, keepdim=True)) / inputs.std(dim=1, keepdim=True)
            orders, _ = model.sample(inputs, num_samples=num_samples, mask=attention_mask)
            edge_acc = model.order_edge_precedence_accuracy(orders[0], targets)
            all_edge_acc.extend(edge_acc.detach().cpu().tolist())
            all_topo_valid.extend((edge_acc >= 1.0).float().detach().cpu().tolist())

    model.train()
    return {
        "order_edge_precedence_accuracy": all_edge_acc,
        "order_topological_validity": all_topo_valid,
    }


def npf_main(args):
    # Start weights and biases
    run = wandb.init(
        # Set the project where this run will be logged
        project="transformer_causal_classifier",
        name=args.run_name,
        # Track hyperparameters and run metadata
        config=vars(args),
    )

    work_dir = Path(args.work_dir).expanduser().resolve()
    synth_data_root = resolve_synth_data_root(args, work_dir)
    models_root = resolve_models_root(args, work_dir)
    results_root = resolve_results_root(args, work_dir)
    data_dir = synth_data_root / args.data_file
    # Get the training and validation datasets
    train_dir = data_dir / "train"
    train_files = list(train_dir.iterdir())
    dataset = MultipleFileDatasetWithPadding(
        [i for i in train_files if i.suffix == ".hdf5"], max_node_num=args.num_nodes
    )
    val_dir = data_dir / "val"
    val_files = list(val_dir.iterdir())
    # Only use like 1000 samples for validation
    val_dataset = MultipleFileDatasetWithPadding(
        [i for i in val_files if i.suffix == ".hdf5"], max_node_num=args.num_nodes
    )

    topo_decoders = {"topo_diffusion", "topo_priority_diffusion"}
    use_bfloat16 = args.topo_bfloat16
    model_dtype = torch.bfloat16 if use_bfloat16 else torch.float32

    TNPD_KWARGS = dict(
        d_model=args.dim_model,
        emb_depth=1,
        dim_feedforward=args.dim_feedforward,
        nhead=args.nhead,
        dropout=args.topo_dropout if args.decoder in topo_decoders else 0.0,
        num_layers_encoder=args.num_layers_encoder,
        num_layers_decoder=(
            args.topo_denoise_layers
            if args.decoder in topo_decoders
            else args.num_layers_decoder
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=model_dtype,
        num_nodes=args.num_nodes,
        n_perm_samples=args.n_perm_samples,
        sinkhorn_iter=args.sinkhorn_iter,
        use_positional_encoding=args.use_positional_encoding,
        num_topo_order_samples=args.num_topo_order_samples,
        ar_hidden_dim=args.ar_hidden_dim,
        topo_num_timesteps=args.topo_num_timesteps,
        topo_sample_N=args.topo_sample_N,
        topo_transition=args.topo_transition,
        topo_reverse=args.topo_reverse,
        topo_reverse_steps=args.topo_reverse_steps,
        topo_beam_size=args.topo_beam_size,
        topo_priority_scale_init=args.topo_priority_scale_init,
    )

    if args.decoder == "probabilistic":
        module = CausalProbabilisticDecoder
    elif args.decoder == "probabilistic_ar":
        if CausalProbabilisticARDecoder is None:
            raise ImportError(
                "CausalProbabilisticARDecoder is not available in the current "
                "causaltransformernp.py. Restore the AR/current file before using "
                "--decoder probabilistic_ar."
            )
        module = CausalProbabilisticARDecoder
    elif args.decoder == "autoregressive":
        module = CsivaDecoder
    elif args.decoder == "transformer":
        module = AviciDecoder
    elif args.decoder == "topo_diffusion":
        module = CausalTopoOrderDiffusion
    elif args.decoder == "topo_priority_diffusion":
        module = CausalPriorityTopoOrderDiffusion
    else:
        raise ValueError(
            "Decoder must be probabilistic, probabilistic_ar, autoregressive, "
            "transformer, topo_diffusion or topo_priority_diffusion"
        )

    model_1d = partial(
        module,
        **TNPD_KWARGS,
    )
    print("Training:", model_1d())

    optimiser = getattr(torch.optim, args.optimizer)
    optimiser_part_init = partial(
        optimiser,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    save_dir = models_root / args.run_name

    # Function to convert dtype objects to serializable format
    def convert_dtype(obj):
        if isinstance(obj, np.dtype):
            return str(obj)

    # Save configs
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w") as f:
        TNPD_KWARGS["module"] = args.decoder
        json.dump(TNPD_KWARGS, f, default=convert_dtype)

    model = model_1d()
    model.to(device=TNPD_KWARGS["device"], dtype=model_dtype)
    optimizer = optimiser_part_init(model.parameters())
    scheduler = (
        torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=args.learning_rate_decay,
        )
        if args.use_learning_rate_decay
        else None
    )

    init_checkpoint = resolve_init_checkpoint(args, models_root=models_root)
    resume_checkpoint = resolve_resume_checkpoint(args, models_root=models_root)
    if init_checkpoint is not None and resume_checkpoint is not None:
        raise ValueError("Use either --init_from_* or --resume_from_*, not both.")

    start_epoch = 0
    if resume_checkpoint is not None:
        start_epoch = load_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_path=resume_checkpoint,
            device=TNPD_KWARGS["device"],
        )
    elif init_checkpoint is not None:
        load_initial_weights(
            model=model,
            checkpoint_path=init_checkpoint,
            device=TNPD_KWARGS["device"],
        )
    trainer = CausalClassifierTrainer(
        train_dataset=dataset,
        validation_dataset=val_dataset,
        test_dataset=val_dataset,
        model=model,
        optimizer=optimizer,
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr_warmup_ratio=args.lr_warmup_ratio, # Should be around 10% of the total steps
        bfloat16=use_bfloat16,
        save_dir=save_dir,
        sample_size_min=args.sample_size_min,
        sample_size_max=args.sample_size_max,
        eval_batch_size=args.eval_batch_size,
        eval_every_epochs=args.eval_every_epochs,
        eval_max_batches=args.eval_max_batches,
        scheduler=scheduler,
        start_epoch=start_epoch,
    )
    trainer.train()
    if args.decoder in topo_decoders:
        metric_dict = trainer.test_single_epoch(
            test_loader=trainer.test_loader,
            metric_dict={},
            calc_metrics=False,
        )
        metric_dict.update(evaluate_topo_order_model(trainer, num_samples=1))
    else:
        metric_dict = trainer.test_single_epoch(
            test_loader=trainer.test_loader,
            metric_dict={},
            calc_metrics=True,
            num_samples=500,
        )

    result_folder = results_root
    result_folder.mkdir(parents=True, exist_ok=True)
    # Save the results
    with open(result_folder / f"{args.run_name}.json", "w") as f:
        json.dump(metric_dict, f)
    pass


if __name__ == "__main__":
    # Log into weights and biases
    wandb.login()

    parser = argparse.ArgumentParser()
    args = retun_default_args(parser)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    npf_main(args)
