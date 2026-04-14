"""
DDPOTrainer wrapper with enhanced logging.

Wraps RLPF's DDPOTrainer to redirect print() to logger.
"""
import sys
import logging
from RLPF-verl_diffusion.verl_diffusion.trainer.ddpo_trainer import DDPOTrainer as BaseDDPOTrainer


class LoggerWriter:
    """Redirect print() to logger."""
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buffer = []
    
    def write(self, message):
        if message.strip():  # Ignore empty lines
            self.logger.log(self.level, message.strip())
    
    def flush(self):
        pass


class DDPOTrainer(BaseDDPOTrainer):
    """DDPOTrainer with logging support."""
    
    def __init__(self, *args, logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger or logging.getLogger("DiffSBDD-DDPO")
        
        # Redirect stdout to logger during training
        self._original_stdout = None
    
    def train(self):
        """Train with logging."""
        # Redirect stdout
        self._original_stdout = sys.stdout
        sys.stdout = LoggerWriter(self.logger)
        
        try:
            super().train()
        finally:
            # Restore stdout
            sys.stdout = self._original_stdout
