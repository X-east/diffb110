import logging

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
    if not torch.cuda.is_available():
        return

    runtime_logger = logger if logger is not None else logging.getLogger("DiffSBDD-DDPO")
    try:
        current_device = device if device is not None else torch.device("cuda", torch.cuda.current_device())
        allocated_mb = torch.cuda.memory_allocated(current_device) / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved(current_device) / (1024 ** 2)
        max_allocated_mb = torch.cuda.max_memory_allocated(current_device) / (1024 ** 2)
        runtime_logger.info(
            "[Memory][%s] %s | device=%s allocated=%.1fMB reserved=%.1fMB max_allocated=%.1fMB",
            component,
            stage,
            current_device,
            allocated_mb,
            reserved_mb,
            max_allocated_mb,
        )
    except Exception:
        runtime_logger.exception("Failed to log CUDA memory for component=%s stage=%s", component, stage)
