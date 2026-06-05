import logging
from typing import Any

import torch

from ddpo.utils.log_control import configure_training_logger


def setup_logging(
    log_dir: str,
    log_level: str = "INFO",
    console_enabled: bool = True,
) -> logging.Logger:
    """Compatibility wrapper over centralized log control."""
    return configure_training_logger(
        log_dir,
        log_level,
        logger_name="DiffSBDD-DDPO",
        console_enabled=console_enabled,
    )


def log_cuda_memory(
    stage: str,
    component: str,
    device: torch.device | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Log CUDA memory statistics through the centralized training logger.

    The helper keeps memory diagnostics decoupled from rollout / actor / model
    logic so those modules only declare *when* to log, not *how* to format or
    route the message.
    """
    runtime_logger = logger if logger is not None else logging.getLogger("DiffSBDD-DDPO")
    if not runtime_logger.isEnabledFor(logging.INFO) or not torch.cuda.is_available():
        return

    try:
        current_device = device if device is not None else torch.device("cuda", torch.cuda.current_device())
        allocated_mb = torch.cuda.memory_allocated(current_device) / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved(current_device) / (1024 ** 2)
        max_allocated_mb = torch.cuda.max_memory_allocated(current_device) / (1024 ** 2)
        max_reserved_mb = torch.cuda.max_memory_reserved(current_device) / (1024 ** 2)
        runtime_logger.info(
            "[Memory][%s] %s | device=%s allocated=%.1fMB reserved=%.1fMB max_allocated=%.1fMB max_reserved=%.1fMB",
            component,
            stage,
            current_device,
            allocated_mb,
            reserved_mb,
            max_allocated_mb,
            max_reserved_mb,
        )
    except Exception:
        runtime_logger.exception("Failed to log CUDA memory for component=%s stage=%s", component, stage)


def reset_cuda_peak_memory(
    stage: str,
    component: str,
    device: torch.device | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Reset CUDA peak memory counters and log the reset point."""
    runtime_logger = logger if logger is not None else logging.getLogger("DiffSBDD-DDPO")
    if not runtime_logger.isEnabledFor(logging.INFO) or not torch.cuda.is_available():
        return

    try:
        current_device = device if device is not None else torch.device("cuda", torch.cuda.current_device())
        torch.cuda.reset_peak_memory_stats(current_device)
        runtime_logger.info("[Memory][%s] %s | reset_peak_memory_stats device=%s", component, stage, current_device)
    except Exception:
        runtime_logger.exception("Failed to reset CUDA peak memory for component=%s stage=%s", component, stage)


def log_dataproto_memory(
    data: Any,
    stage: str,
    component: str,
    logger: logging.Logger | None = None,
    max_keys: int = 16,
) -> None:
    """Log tensor devices and approximate tensor bytes in a DataProto-like object."""
    runtime_logger = logger if logger is not None else logging.getLogger("DiffSBDD-DDPO")
    if not runtime_logger.isEnabledFor(logging.INFO):
        return

    batch = getattr(data, "batch", None)
    if batch is None:
        runtime_logger.info("[Memory][%s] %s | DataProto batch=None", component, stage)
        return

    totals: dict[str, int] = {}
    details: list[str] = []
    try:
        keys = list(batch.keys())
        for idx, key in enumerate(keys):
            tensor = batch[key]
            if not isinstance(tensor, torch.Tensor):
                continue
            device = str(tensor.device)
            nbytes = int(tensor.element_size() * tensor.numel())
            totals[device] = totals.get(device, 0) + nbytes
            if idx < max_keys:
                shape = "x".join(str(dim) for dim in tensor.shape)
                details.append(f"{key}:{shape}:{device}:{nbytes / (1024 ** 2):.1f}MB")

        totals_text = ", ".join(
            f"{device}={nbytes / (1024 ** 2):.1f}MB"
            for device, nbytes in sorted(totals.items())
        )
        detail_text = "; ".join(details)
        omitted = max(0, len(keys) - max_keys)
        if omitted:
            detail_text = f"{detail_text}; ... +{omitted} keys"
        runtime_logger.info(
            "[Memory][%s] %s | DataProto tensors total[%s] keys[%s]",
            component,
            stage,
            totals_text or "none",
            detail_text or "none",
        )
    except Exception:
        runtime_logger.exception("Failed to log DataProto memory for component=%s stage=%s", component, stage)
