import argparse
from argparse import Namespace
import os
from pathlib import Path
from time import monotonic
from typing import Any, Iterable
import warnings

import torch
import pytorch_lightning as pl
import yaml
import numpy as np

from lightning_modules import LigandPocketDDPM


def configure_cuda_runtime(args):
    """Configure rank-local CUDA state before module construction."""
    if not torch.cuda.is_available():
        return

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        device_index = int(local_rank) % max(torch.cuda.device_count(), 1)
        torch.cuda.set_device(device_index)
        print(
            "Configured local CUDA device before model init: "
            f"LOCAL_RANK={local_rank}, cuda:{device_index}",
            flush=True,
        )

    # Several DiffSBDD EGNN blocks call self.to(device) inside __init__.
    # In DDP this can make every rank touch cuda:0 before Lightning places the
    # module on the correct device. Build on CPU first, then let Lightning move
    # the full module to each rank-local GPU.
    if int(getattr(args, "gpus", 0) or 0) > 1 and hasattr(args, "egnn_params"):
        previous_device = getattr(args.egnn_params, "device", None)
        args.egnn_params.device = "cpu"
        print(
            "Using CPU module initialization for DDP: "
            f"egnn_params.device {previous_device!r} -> 'cpu'",
            flush=True,
        )


def namespace_get(config: Any, key: str, default: Any = None) -> Any:
    """Return one config value from a dict/Namespace-like object."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def tensor_to_float(value: Any) -> float | None:
    """Convert scalar tensors and numbers to float for log printing."""
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, (float, int)):
        return float(value)
    return None


def format_metrics(metrics: dict[str, Any], names: Iterable[str]) -> str:
    """Format selected scalar metrics for compact terminal output."""
    parts = []
    for name in names:
        value = tensor_to_float(metrics.get(name))
        if value is not None:
            parts.append(f"{name}={value:.4g}")
    return " ".join(parts)


class LocalTrainingProgressCallback(pl.callbacks.Callback):
    """Print rank-0 training progress and epoch metrics to stdout."""

    DEFAULT_STEP_METRICS = (
        "loss",
        "edge_loss_weighted",
        "bond_length_loss_weighted",
        "edge_smoke_loss",
        "edge_smoke_acc",
        "edge_smoke_real_bond_acc",
        "bond_length_loss",
    )
    DEFAULT_EPOCH_METRICS = (
        "loss/train",
        "loss/val",
        "edge_loss_weighted/train",
        "edge_loss_weighted/val",
        "bond_length_loss_weighted/train",
        "bond_length_loss_weighted/val",
        "edge_smoke_loss/train",
        "edge_smoke_loss/val",
        "edge_smoke_acc/train",
        "edge_smoke_acc/val",
        "edge_smoke_real_bond_acc/train",
        "edge_smoke_real_bond_acc/val",
        "bond_length_loss/train",
        "bond_length_loss/val",
        "error_t_lig/train",
        "error_t_lig/val",
        "loss_0/train",
        "loss_0/val",
        "kl_prior/train",
        "kl_prior/val",
    )

    def __init__(
            self,
            train_every_n_batches: int = 500,
            epoch_metrics: Iterable[str] | None = None,
            step_metrics: Iterable[str] | None = None,
    ) -> None:
        self.train_every_n_batches = max(0, int(train_every_n_batches))
        self.epoch_metrics = tuple(epoch_metrics or self.DEFAULT_EPOCH_METRICS)
        self.step_metrics = tuple(step_metrics or self.DEFAULT_STEP_METRICS)
        self._start_time = monotonic()

    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
    ) -> None:
        """Print a compact train-step line at a fixed batch interval."""
        del pl_module, batch
        if self.train_every_n_batches <= 0 or not trainer.is_global_zero:
            return
        if batch_idx % self.train_every_n_batches != 0:
            return
        metrics = outputs if isinstance(outputs, dict) else {}
        metric_text = format_metrics(metrics, self.step_metrics)
        elapsed_min = (monotonic() - self._start_time) / 60.0
        print(
            "[train-progress] "
            f"epoch={trainer.current_epoch + 1}/{trainer.max_epochs} "
            f"batch_idx={batch_idx} "
            f"global_step={trainer.global_step} "
            f"elapsed_min={elapsed_min:.1f} "
            f"{metric_text}",
            flush=True,
        )

    def on_train_epoch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
    ) -> None:
        """Print train epoch metrics collected by Lightning."""
        del pl_module
        self._print_epoch_metrics(trainer, "train-epoch-end")

    def on_validation_epoch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
    ) -> None:
        """Print validation metrics collected by Lightning."""
        del pl_module
        self._print_epoch_metrics(trainer, "validation-epoch-end")

    def _print_epoch_metrics(self, trainer: pl.Trainer, label: str) -> None:
        """Print selected callback metrics on rank 0."""
        if not trainer.is_global_zero:
            return
        metrics = dict(trainer.callback_metrics)
        metric_text = format_metrics(metrics, self.epoch_metrics)
        elapsed_min = (monotonic() - self._start_time) / 60.0
        print(
            f"[{label}] "
            f"epoch={trainer.current_epoch + 1}/{trainer.max_epochs} "
            f"global_step={trainer.global_step} "
            f"elapsed_min={elapsed_min:.1f} "
            f"{metric_text}",
            flush=True,
        )


def build_training_logger(args):
    """Build a local CSV logger by default, with opt-in WandB support."""
    logging_config = namespace_get(args, "local_logging", None)
    backend = str(namespace_get(
        logging_config,
        "backend",
        namespace_get(args, "logger_backend", "csv"),
    )).lower()
    if backend in {"csv", "local_csv", "local"}:
        version = namespace_get(logging_config, "version", "metrics")
        logger = pl.loggers.CSVLogger(
            save_dir=args.logdir,
            name=args.run_name,
            version=version,
        )
        print(
            "Using local CSV logger: "
            f"{Path(logger.log_dir, 'metrics.csv')}",
            flush=True,
        )
        return logger
    if backend == "wandb":
        raise ValueError(
            "WandB logging is disabled by project rule. "
            "Use local CSV logging instead.")
    if backend in {"none", "false", "disabled"}:
        print("Lightning logger disabled by config.", flush=True)
        return False
    raise ValueError(
        "Unsupported logger backend. Use 'csv' for formal training. "
        f"Got: {backend!r}")


def build_progress_callback(args) -> LocalTrainingProgressCallback:
    """Create the rank-0 stdout progress callback from config."""
    logging_config = namespace_get(args, "local_logging", None)
    train_every_n_batches = namespace_get(
        logging_config,
        "print_train_every_n_batches",
        namespace_get(args, "print_train_every_n_batches", 500),
    )
    return LocalTrainingProgressCallback(
        train_every_n_batches=int(train_every_n_batches),
    )


def filter_shape_compatible_state_dict(module, state_dict):
    """Return checkpoint tensors whose names and shapes match the target module."""
    target_state = module.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        target_value = target_state.get(key)
        if target_value is None:
            skipped.append(key)
            continue
        if tuple(target_value.shape) != tuple(value.shape):
            skipped.append(key)
            continue
        compatible[key] = value
    return compatible, skipped


def validate_xhe_warm_start_policy(args, state_dict, checkpoint_path):
    """Prevent accidental x/h-only warm starts for the xhe mainline."""
    if getattr(args, 'diffusion_mode', 'xh') != 'xhe':
        return
    policy = getattr(args, 'xhe_warm_start_policy', 'xhe_only')
    if policy == 'unsafe_any':
        warnings.warn(
            "xhe_warm_start_policy='unsafe_any' bypasses the xhe checkpoint "
            "guard and should be used only for explicit ablations.")
        return
    if policy == 'scratch':
        raise ValueError(
            "diffusion_mode='xhe' with xhe_warm_start_policy='scratch' "
            "must not set warm_start_checkpoint or --warm-start.")
    if policy != 'xhe_only':
        raise ValueError(
            "xhe_warm_start_policy must be one of: scratch, xhe_only, unsafe_any")
    marker_fragments = (
        'chemical_edge_embedding',
        'chemical_message_embedding',
        'edge_time_mlp',
        'chemical_edge_decoder',
    )
    if not any(any(fragment in key for fragment in marker_fragments)
               for key in state_dict):
        raise ValueError(
            "Refusing to warm-start diffusion_mode='xhe' from a checkpoint "
            f"without xhe edge-head parameters: {checkpoint_path}. Use the "
            "new scratch xhe configs, resume from an xhe checkpoint, or set "
            "xhe_warm_start_policy='unsafe_any' for an explicit ablation.")


def merge_args_and_yaml(args, config_dict):
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if key in arg_dict:
            warnings.warn(f"Command line argument '{key}' (value: "
                          f"{arg_dict[key]}) will be overwritten with value "
                          f"{value} provided in the config file.")
        if isinstance(value, dict):
            arg_dict[key] = Namespace(**value)
        else:
            arg_dict[key] = value

    return args


def merge_configs(config, resume_config):
    for key, value in resume_config.items():
        if isinstance(value, Namespace):
            value = value.__dict__
        if key in config and config[key] != value:
            warnings.warn(f"Config parameter '{key}' (value: "
                          f"{config[key]}) will be overwritten with value "
                          f"{value} from the checkpoint.")
        config[key] = value
    return config


# ------------------------------------------------------------------------------
# Training
# ______________________________________________________________________________
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--warm-start', type=str, default=None)
    args = p.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    assert 'resume' not in config

    # Get main config
    ckpt_path = None if args.resume is None else Path(args.resume)
    if args.resume is not None:
        resume_config = torch.load(
            ckpt_path, map_location=torch.device('cpu'))['hyper_parameters']

        config = merge_configs(config, resume_config)

    args = merge_args_and_yaml(args, config)
    configure_cuda_runtime(args)
    if getattr(args, 'seed', None) is not None:
        pl.seed_everything(int(args.seed), workers=True)

    out_dir = Path(args.logdir, args.run_name)
    histogram_file = Path(args.datadir, 'size_distribution.npy')
    histogram = np.load(histogram_file).tolist()
    pl_module = LigandPocketDDPM(
        outdir=out_dir,
        dataset=args.dataset,
        datadir=args.datadir,
        batch_size=args.batch_size,
        lr=args.lr,
        egnn_params=args.egnn_params,
        diffusion_params=args.diffusion_params,
        num_workers=args.num_workers,
        augment_noise=args.augment_noise,
        augment_rotation=args.augment_rotation,
        clip_grad=args.clip_grad,
        eval_epochs=args.eval_epochs,
        eval_params=args.eval_params,
        visualize_sample_epoch=args.visualize_sample_epoch,
        visualize_chain_epoch=args.visualize_chain_epoch,
        auxiliary_loss=args.auxiliary_loss,
        loss_params=args.loss_params,
        mode=args.mode,
        node_histogram=histogram,
        pocket_representation=args.pocket_representation,
        virtual_nodes=args.virtual_nodes,
        diffusion_mode=getattr(args, 'diffusion_mode', 'xh'),
        bond_diffusion=getattr(args, 'bond_diffusion', None),
    )
    warm_start_checkpoint = args.warm_start or getattr(
        args, 'warm_start_checkpoint', None)
    if warm_start_checkpoint is not None:
        checkpoint = torch.load(
            warm_start_checkpoint, map_location=torch.device('cpu'))
        state_dict = checkpoint.get('state_dict', checkpoint)
        validate_xhe_warm_start_policy(args, state_dict, warm_start_checkpoint)
        state_dict, skipped_shape_keys = filter_shape_compatible_state_dict(
            pl_module,
            state_dict,
        )
        incompatible = pl_module.load_state_dict(state_dict, strict=False)
        print(
            "Warm-started from "
            f"{warm_start_checkpoint}: "
            f"missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}, "
            f"skipped_shape_or_missing={len(skipped_shape_keys)}")
        if incompatible.missing_keys:
            print("Missing keys sample:", incompatible.missing_keys[:20])
        if incompatible.unexpected_keys:
            print("Unexpected keys sample:", incompatible.unexpected_keys[:20])
        if skipped_shape_keys:
            print("Skipped incompatible keys sample:", skipped_shape_keys[:20])

    logger = build_training_logger(args)

    checkpoint_dir = Path(out_dir, 'checkpoints')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-model-epoch={epoch:04d}",
        monitor="loss/val",
        save_top_k=1,
        save_last=True,
        mode="min",
    )
    callbacks = [checkpoint_callback, build_progress_callback(args)]

    checkpoint_every_n_epochs = getattr(args, 'checkpoint_every_n_epochs', None)
    if checkpoint_every_n_epochs is not None:
        checkpoint_every_n_epochs = int(checkpoint_every_n_epochs)
        if checkpoint_every_n_epochs <= 0:
            raise ValueError("checkpoint_every_n_epochs must be positive")
        periodic_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch={epoch:04d}",
            save_top_k=-1,
            every_n_epochs=checkpoint_every_n_epochs,
            save_on_train_epoch_end=True,
        )
        callbacks.append(periodic_checkpoint_callback)

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=args.enable_progress_bar,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accumulate_grad_batches=getattr(args, 'accumulate_grad_batches', 1),
        log_every_n_steps=int(namespace_get(
            namespace_get(args, "local_logging", None),
            "log_every_n_steps",
            50,
        )),
        accelerator='gpu', devices=args.gpus,
        strategy=('ddp' if args.gpus > 1 else None)
    )

    trainer.fit(model=pl_module, ckpt_path=ckpt_path)
