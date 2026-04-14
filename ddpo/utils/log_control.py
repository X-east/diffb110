"""Logging control utilities for training runtime.

This module centralizes output policy:
- Console: concise, high-signal messages only
- File: full debug trace for diagnosis

Use `logger.info(..., extra={"is_console": True})` for messages that should
be visible in terminal.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path


class _ConsoleSignalFilter(logging.Filter):
    """Allow warnings/errors and explicitly marked info records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return bool(getattr(record, "is_console", False))


def configure_training_logger(
    output_dir: str,
    log_level: str = "INFO",
    logger_name: str = "DiffSBDD-DDPO",
    console_enabled: bool = True,
) -> logging.Logger:
    """Configure a logger with split console/file policy.

    Args:
        output_dir: Training run output directory.
        log_level: Base logger level.
        logger_name: Logger name.

    Returns:
        Configured logger instance.
    """
    logs_dir = Path(output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_log = logs_dir / f"training_{timestamp}.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid duplicate handlers when setup is called repeatedly.
    if logger.handlers:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    logger.propagate = False

    file_handler = logging.FileHandler(training_log, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.addFilter(_ConsoleSignalFilter())

    logger.addHandler(file_handler)
    if console_enabled:
        logger.addHandler(console_handler)

    # Export paths for modules that need dedicated sinks.
    os.environ["DIFFSBDD_LOG_DIR"] = str(logs_dir)
    os.environ["DIFFSBDD_TRAINING_LOG"] = str(training_log)

    if console_enabled:
        logger.info(
            f"Logging to: {training_log}",
            extra={"is_console": True},
        )

    return logger


def console_info(logger: logging.Logger, message: str) -> None:
    """Log an info message that is visible in terminal."""
    logger.info(message, extra={"is_console": True})


def get_chem_warnings_log_path() -> str:
    """Return chemistry warning sink path under logs directory."""
    logs_dir = os.environ.get("DIFFSBDD_LOG_DIR", ".")
    return str(Path(logs_dir) / "chem_warnings.log")
