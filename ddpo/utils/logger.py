"""
Training Logger for DiffSBDD-RLPF.

Provides unified logging to:
- Console (real-time)
- Log file (detailed)
- CSV file (metrics for analysis)
"""

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class TrainingLogger:
    """
    Unified logger for RL training.
    
    Logs to:
    - Console: Real-time progress
    - .log file: Detailed text logs
    - .csv file: Metrics for plotting
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "diffsbdd-ddpo",
        log_level: str = "INFO",
    ):
        """
        Initialize the training logger.
        
        Args:
            log_dir: Directory to save log files
            experiment_name: Name for this experiment
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        # Setup file paths
        self.log_file = self.log_dir / f"training_{self.timestamp}.log"
        self.csv_file = self.log_dir / f"metrics_{self.timestamp}.csv"
        
        # Setup Python logger
        self.logger = self._setup_logger(log_level)
        
        # Setup CSV writer
        self.csv_writer = None
        self.csv_file_handle = None
        self.csv_headers_written = False
        
        self.logger.info(f"Training logger initialized")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"CSV file: {self.csv_file}")
    
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Setup Python logger with file and console handlers."""
        logger = logging.getLogger(f"DiffSBDD-DDPO-{self.timestamp}")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Log format
        log_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # File handler (detailed)
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        
        # Console handler (concise)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration at start of training."""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING CONFIGURATION")
        self.logger.info("=" * 60)
        self._log_dict(config, indent=0)
        self.logger.info("=" * 60)
    
    def _log_dict(self, d: Dict, indent: int = 0):
        """Recursively log dictionary."""
        prefix = "  " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                self.logger.info(f"{prefix}{k}:")
                self._log_dict(v, indent + 1)
            else:
                self.logger.info(f"{prefix}{k}: {v}")
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log start of epoch."""
        self.logger.info("-" * 40)
        self.logger.info(f"EPOCH {epoch + 1}/{total_epochs}")
        self.logger.info("-" * 40)
    
    def log_metrics(
        self,
        epoch: int,
        batch: int,
        metrics: Dict[str, float],
        extra_info: Optional[Dict[str, Any]] = None
    ):
        """
        Log training metrics.
        
        Args:
            epoch: Current epoch
            batch: Current batch within epoch
            metrics: Dictionary of metric name -> value
            extra_info: Additional info to log (not saved to CSV)
        """
        # Log to console/file
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"[E{epoch+1} B{batch}] {metrics_str}")
        
        if extra_info:
            for k, v in extra_info.items():
                self.logger.debug(f"  {k}: {v}")
        
        # Log to CSV
        self._write_csv(epoch, batch, metrics)
    
    def _write_csv(self, epoch: int, batch: int, metrics: Dict[str, float]):
        """Write metrics to CSV file."""
        # Open file if not already open
        if self.csv_file_handle is None:
            self.csv_file_handle = open(self.csv_file, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file_handle)
        
        # Write headers on first call
        if not self.csv_headers_written:
            headers = ['timestamp', 'epoch', 'batch'] + list(metrics.keys())
            self.csv_writer.writerow(headers)
            self.csv_headers_written = True
        
        # Write data row
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, epoch, batch] + list(metrics.values())
        self.csv_writer.writerow(row)
        self.csv_file_handle.flush()  # Ensure data is written
    
    def log_checkpoint(self, path: str, metrics: Dict[str, float]):
        """Log checkpoint save."""
        self.logger.info(f"Checkpoint saved: {path}")
        self.logger.info(f"  Metrics: {metrics}")
    
    def log_summary(
        self,
        total_epochs: int,
        best_reward: float,
        final_metrics: Dict[str, float]
    ):
        """Log training summary at end."""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Total epochs: {total_epochs}")
        self.logger.info(f"Best reward: {best_reward:.4f}")
        self.logger.info("Final metrics:")
        for k, v in final_metrics.items():
            self.logger.info(f"  {k}: {v:.4f}")
        self.logger.info("=" * 60)
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"CSV file: {self.csv_file}")
    
    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)
    
    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)
    
    def close(self):
        """Close file handles."""
        if self.csv_file_handle:
            self.csv_file_handle.close()
            self.csv_file_handle = None
        
        # Close logger handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
