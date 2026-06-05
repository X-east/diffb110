"""
Stub module to replace wandb when it's not installed.
This allows DiffSBDD to import without errors while using local logging instead.

All logs are saved to the project root's log/ directory.

Usage:
    import sys
    from ddpo.utils import wandb_stub
    sys.modules['wandb'] = wandb_stub
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

# Get project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "log"


class Run:
    """Stub for wandb.Run that logs to local files."""
    
    def __init__(self, name: str = None, project: str = "diffsbdd", config: dict = None):
        self.name = name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.id = self.name
        self.project = project
        self.config = config or {}
        
        # Create run directory
        self.dir = LOG_DIR / project / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self._metrics: List[Dict] = []
        self._step = 0
        self._metrics_file = self.dir / "metrics.csv"
        self._metrics_json = self.dir / "metrics.json"
        self._images_log = self.dir / "images.json"
        self._videos_log = self.dir / "videos.json"
        
        # Save config
        if config:
            with open(self.dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, default=str)
        
        print(f"[wandb_stub] Run initialized: {self.dir}")
    
    def log(self, data: Dict[str, Any], step: int = None, commit: bool = True):
        """Log metrics to local files."""
        if step is None:
            step = self._step
            self._step += 1
        
        # Process data - extract values from wandb objects
        processed = {"step": step, "timestamp": datetime.now().isoformat()}
        for key, value in data.items():
            if isinstance(value, (int, float, str, bool)):
                processed[key] = value
            elif isinstance(value, list):
                # Handle lists of images/videos
                for i, item in enumerate(value):
                    if hasattr(item, '_path'):
                        processed[f"{key}_{i}_path"] = item._path
        
        self._metrics.append(processed)
        
        # Print to console
        metrics_str = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in data.items() 
            if isinstance(v, (int, float))
        )
        if metrics_str:
            print(f"[wandb_stub] Step {step}: {metrics_str}")
        
        if commit:
            self._write_metrics()
    
    def _write_metrics(self):
        """Write metrics to CSV and JSON files."""
        if not self._metrics:
            return
        
        # Write JSON
        with open(self._metrics_json, "w", encoding="utf-8") as f:
            json.dump(self._metrics, f, indent=2, default=str)
        
        # Write CSV
        all_keys = set()
        for record in self._metrics:
            all_keys.update(record.keys())
        
        fieldnames = ["step", "timestamp"] + sorted(
            k for k in all_keys if k not in ["step", "timestamp"]
        )
        
        with open(self._metrics_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._metrics)
    
    def finish(self):
        """Finish the run and save final metrics."""
        self._write_metrics()
        print(f"[wandb_stub] Run finished. Logs saved to: {self.dir}")


class Api:
    """Stub for wandb.Api"""
    pass


class _ImageStub:
    """Stub for wandb.Image"""
    def __init__(self, data, caption=None):
        self._data = data
        self._caption = caption
        self._path = str(data) if isinstance(data, (str, Path)) else None


class _VideoStub:
    """Stub for wandb.Video"""
    def __init__(self, data, caption=None):
        self._data = data
        self._caption = caption
        self._path = str(data) if isinstance(data, (str, Path)) else None


# Global run instance
run: Optional[Run] = None


def init(
    project: str = "diffsbdd",
    name: str = None,
    config: dict = None,
    **kwargs
) -> Run:
    """Initialize a new run. If already initialized, return existing run."""
    global run
    # If already initialized, just update config if provided
    if run is not None:
        print(f"[wandb_stub] Run already initialized, reusing: {run.dir}")
        if config:
            run.config.update(config)
        return run
    run = Run(name=name, project=project, config=config)
    return run


def log(data: Dict[str, Any], step: int = None, commit: bool = True):
    """Log metrics to the current run."""
    global run
    if run is None:
        run = Run()
    run.log(data, step=step, commit=commit)


def finish():
    """Finish the current run."""
    global run
    if run is not None:
        run.finish()
        run = None


def watch(*args, **kwargs):
    """Stub for wandb.watch (model watching)."""
    pass


def save(*args, **kwargs):
    """Stub for wandb.save."""
    pass


def define_metric(*args, **kwargs):
    """Stub for wandb.define_metric."""
    pass


def Image(data, caption=None, **kwargs):
    """Create an image object."""
    return _ImageStub(data, caption)


def Video(data, caption=None, **kwargs):
    """Create a video object."""
    return _VideoStub(data, caption)


def Table(*args, **kwargs):
    """Stub for wandb.Table."""
    return None


def Histogram(*args, **kwargs):
    """Stub for wandb.Histogram."""
    return None


def Audio(*args, **kwargs):
    """Stub for wandb.Audio."""
    return None


def Html(*args, **kwargs):
    """Stub for wandb.Html."""
    return None


# Constants
START_TIME = 0
