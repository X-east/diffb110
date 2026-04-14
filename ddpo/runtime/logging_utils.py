import logging

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
