"""
Main entry point for DiffSBDD-RLPF DDPO training.

This script integrates DiffSBDD (E(n)-equivariant diffusion model) with
RLPF's DDPO training framework for optimizing molecular generation
towards the RE-CmeB protein target.

Usage:
    python ddpo/main_diffsbdd.py --config_path ddpo/configs/diffsbdd_ddpo.yaml
    python ddpo/main_diffsbdd.py --config_path ddpo/configs/diffsbdd_ddpo.yaml --log_level DEBUG

Requirements:
    - Task 4: DiffSBDDModel (model wrapper)
    - Task 8-10: Reward functions (DockingReward, DruglikenessReward, CompositeReward)
    - Task 11: DiffSBDDFilter (molecule validity filter)
    - Task 12: Checkpoint (all reward tests passing)
"""

import argparse
import logging
import os
import signal
import sys
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import ray
import yaml

# Disable output buffering for immediate console output
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "RLPF-verl_diffusion"))
sys.path.insert(0, str(PROJECT_ROOT / "DiffSBDD"))

# Activate wandb_stub for logging (MUST be before other imports)
import ddpo.utils  # This registers wandb_stub

import wandb  # Now this will use wandb_stub

# Import RLPF components
from verl_diffusion.trainer.ddpo_trainer import DDPOTrainer
from verl_diffusion.worker.actor.edm_actor import EDMActor
from verl_diffusion.protocol import DataProto, TensorDict

# Import DiffSBDD constants
from constants import dataset_params

# Import our custom components
from ddpo.integration import DiffSBDDModel
from ddpo.rewards import DockingReward, DruglikenessReward, CompositeReward
from ddpo.utils import DiffSBDDFilter
from ddpo.utils.log_control import configure_training_logger
from ddpo.utils.suppress_warnings import redirect_stderr_to_log


# Global flag for graceful shutdown
_shutdown_requested = False


# =============================================================================
# Signal handlers for graceful shutdown
# =============================================================================

def signal_handler(signum, frame):
    """Handle interrupt signals for graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\nShutdown requested. Finishing current batch...")


# =============================================================================
# Logging setup
# =============================================================================

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


# =============================================================================
# Dataloader for DiffSBDD
# =============================================================================

class DiffSBDDDataLoader:
    """
    DataLoader for DiffSBDD DDPO training.
    
    Generates batches of node counts for molecular generation.
    Unlike EDMDataLoader which samples from a learned distribution,
    this uses fixed or configurable node counts for drug-like molecules.
    """
    
    def __init__(
        self,
        config: dict,
        dataset_info: dict,
        device: str = "cuda",
        num_batches: int = None,
    ):
        """
        Initialize the dataloader.
        
        Args:
            config: Configuration dictionary
            dataset_info: Dataset info from DiffSBDD constants
            device: Device to put tensors on
            num_batches: Number of batches per epoch
        """
        self.config = config
        self.dataset_info = dataset_info
        self.device = device
        self.num_batches = num_batches or config["dataloader"].get("epoches", config["dataloader"].get("epochs", 100))
        
        # Sampling parameters
        self.group_sample = config["dataloader"].get("sample_group_size", 4)
        self.each_prompt_sample = config["dataloader"].get("each_prompt_sample", 32)
        self.batch_size = self.group_sample * self.each_prompt_sample
        
        # Node count range for drug-like molecules (typically 10-50 atoms)
        self.min_nodes = config["dataloader"].get("min_nodes", 10)
        self.max_nodes = config["dataloader"].get("max_nodes", 50)
        # max_n_nodes must match max_nodes so masks are correctly sized
        self.max_n_nodes = config["dataloader"].get("max_n_nodes",
                           config["dataloader"].get("max_nodes",
                           dataset_info.get("max_n_nodes", 50)))
        
        self.current_batch = 0
    
    def __iter__(self):
        self.current_batch = 0
        return self
    
    def __len__(self):
        # Each Lightning epoch = one generate-then-train cycle.
        # The dataloader yields exactly 1 batch per epoch.
        # Total number of cycles is controlled by Trainer(max_epochs=epochs).
        return 1
    
    def __next__(self):
        if self.current_batch >= 1:
            raise StopIteration
        self.current_batch += 1
        return self.generate_batch()
    
    def generate_batch(self):
        """Generate a batch of prompts (node counts)."""
        # Sample node counts for each group
        nodesxsample = torch.randint(
            self.min_nodes, self.max_nodes + 1,
            (self.group_sample,),
            device=self.device
        )
        
        # Create group indices
        group_index = torch.arange(self.group_sample, device=self.device)
        group_index = group_index.repeat_interleave(self.each_prompt_sample)
        
        # Repeat node counts for each sample in the group
        nodesxsample = nodesxsample.repeat_interleave(self.each_prompt_sample)
        
        # Create batch data
        tensors = {
            "nodesxsample": nodesxsample,
            "group_index": group_index,
        }
        
        meta_info = {
            "max_n_nodes": self.max_n_nodes,
            "condition": False,
        }
        
        return DataProto(
            batch=TensorDict(tensors, batch_size=[self.batch_size]),
            meta_info=meta_info
        )
    
    def reset(self):
        self.current_batch = 0


# =============================================================================
# Rollout for DiffSBDD
# =============================================================================

class DiffSBDDRollout:
    """
    Rollout worker for DiffSBDD model.
    
    Generates molecular samples using DiffSBDDModel and records
    trajectories for DDPO training.
    """
    
    def __init__(self, model: DiffSBDDModel, config: dict):
        """
        Initialize the rollout worker.
        
        Args:
            model: DiffSBDDModel instance
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.timestep = config["model"].get("time_step", 500)
    
    def generate_minibatch(self, prompts: DataProto) -> DataProto:
        """
        Generate a minibatch of molecular samples.
        
        Args:
            prompts: DataProto containing nodesxsample and group_index
            
        Returns:
            DataProto containing generated samples with trajectories
        """
        batch_size = prompts.batch.batch_size[0]
        max_n_nodes = prompts.meta_info["max_n_nodes"]
        nodesxsample = prompts.batch["nodesxsample"]
        device = nodesxsample.device
        
        # Get masks
        node_mask, edge_mask = self.model.get_mask(
            nodesxsample, batch_size, max_n_nodes
        )
        node_mask = node_mask.to(device)
        edge_mask = edge_mask.to(device)
        
        # Sample from model
        x, h, latents, logps, timesteps, mus, sigmas = self.model.sample(
            n_samples=batch_size,
            n_nodes=max_n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            timestep=self.timestep
        )
        
        # ----------------------------------------------------------------
        # Trajectory alignment for EDMActor:
        #
        # model.sample() returns:
        #   latents : list of T+2 tensors  [z_T, z_{T-1}, ..., z_0, xh_final]
        #   logps   : list of T+1 tensors  [lp_{T-1}, ..., lp_0, lp_final]
        #   timesteps: list of T+1 ints    [T-1, T-2, ..., 0]
        #
        # EDMActor.update_policy slices:
        #   samples["latents"]      = latents[:, :T+1]   (indices 0..T)
        #   samples["next_latents"] = latents[:, 1:]     (indices 1..T+1)
        #   samples["logps"]        = logps              (T+1 entries)
        #   samples["timesteps"]    = timesteps          (T+1 entries)
        # For step j: loss uses latents[:,j], next_latents[:,j], logps[:,j],
        # timesteps[:,j] — all T+1 aligned.
        # We stack latents as-is (T+2); the actor slices [:T+1] internally.
        # ----------------------------------------------------------------

        latents_stacked  = torch.stack(latents, dim=1)   # [B, T+2, N, D]
        logps_stacked    = torch.stack(logps,   dim=1)   # [B, T+1]

        # timesteps: list of T+1 integer s-values [T-1, ..., 0]
        timesteps_tensor = torch.tensor(timesteps, dtype=torch.long, device=device)  # [T+1]
        expanded_timesteps = timesteps_tensor.unsqueeze(0).expand(batch_size, -1).clone()  # [B, T+1]

        # Sanity-check shapes
        assert logps_stacked.shape[1] == expanded_timesteps.shape[1], (
            f"logps length {logps_stacked.shape[1]} != "
            f"timesteps length {expanded_timesteps.shape[1]}"
        )

        # h format: DiffSBDDModel returns dict with 'categorical' key
        if isinstance(h, dict):
            h_categorical = h["categorical"]
        else:
            h_categorical = h
        
        # Create batch data
        batch_data = {
            "x": x,
            "categorical": h_categorical,
            "latents": latents_stacked,
            "logps": logps_stacked,
            "nodesxsample": nodesxsample,
            "timesteps": expanded_timesteps,
            "group_index": prompts.batch["group_index"],
        }
        
        return DataProto(
            batch=TensorDict(batch_data, batch_size=batch_size),
            meta_info=prompts.meta_info.copy()
        )


# =============================================================================
# Stability-aware reward wrapper
# =============================================================================

class StabilityAwareReward:
    """
    Wrapper that adds 'stability' field to reward output.
    
    DDPOTrainer expects samples.batch['stability'] for logging.
    This wrapper computes molecule stability using DiffSBDD's validity check.
    """
    
    def __init__(self, base_reward, dataset_info: dict):
        """
        Initialize the wrapper.
        
        Args:
            base_reward: Base reward function (e.g., CompositeReward)
            dataset_info: Dataset info for stability computation
        """
        self.base_reward = base_reward
        self.dataset_info = dataset_info
    
    def calculate_rewards(self, data: DataProto) -> DataProto:
        """
        Calculate rewards and add stability metric.
        
        Args:
            data: DataProto containing molecule data
            
        Returns:
            DataProto with 'rewards' and 'stability' tensors
        """
        # Calculate base rewards
        data = self.base_reward.calculate_rewards(data)
        
        # Add stability metric if not present
        if 'stability' not in data.batch:
            batch_size = data.batch['x'].shape[0]
            device = data.batch['x'].device
            
            # Compute stability based on validity
            # For DiffSBDD, we use the validity check from build_molecule
            stability = self._compute_stability(data)
            data.batch['stability'] = stability.to(device)
        
        return data
    
    def _compute_stability(self, data: DataProto) -> torch.Tensor:
        """
        Compute molecule stability.
        
        Uses RDKit sanitization as a proxy for stability.
        """
        from rdkit import Chem
        from analysis.molecule_builder import build_molecule
        
        x = data.batch['x']
        h = data.batch['categorical']
        nodesxsample = data.batch['nodesxsample']
        
        batch_size = x.shape[0]
        stability = torch.zeros(batch_size)
        
        for i in range(batch_size):
            n_nodes = nodesxsample[i].item()
            pos_i = x[i, :n_nodes].cpu().numpy()
            atom_types_i = h[i, :n_nodes].argmax(dim=-1).cpu().numpy()
            
            try:
                with redirect_stderr_to_log():
                    mol = build_molecule(
                        positions=torch.from_numpy(pos_i),
                        atom_types=torch.from_numpy(atom_types_i),
                        dataset_info=self.dataset_info,
                        add_coords=True,
                        use_openbabel=False  # Disable OpenBabel to avoid plugin issues
                    )

                if mol is not None:
                    try:
                        with redirect_stderr_to_log():
                            Chem.SanitizeMol(mol)
                        stability[i] = 1.0
                    except Exception:
                        stability[i] = 0.0
            except Exception:
                stability[i] = 0.0
        
        return stability
    
    def __repr__(self) -> str:
        return f"StabilityAwareReward({self.base_reward})"


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DDPO Training for DiffSBDD molecular generation"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="ddpo/configs/diffsbdd_ddpo.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save checkpoints (overrides config)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint path for resuming"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Load and validate configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required parameters are missing
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ["model", "pocket", "reward", "dataloader", "train"]
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")
    
    # Validate required parameters
    required_params = [
        ("model", "checkpoint"),
        ("pocket", "pdb_path"),
        ("train", "batch_size"),
        ("train", "learning_rate"),
    ]
    
    missing_params = []
    for section, param in required_params:
        if section not in config or param not in config[section]:
            missing_params.append(f"{section}.{param}")
    
    if missing_params:
        raise ValueError(f"Missing required config parameters: {missing_params}")
    
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Main function
# =============================================================================

def main():
    """Main training function."""
    global _shutdown_requested
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Setup output directory
    output_dir = args.save_path or config.get("output", {}).get("dir", "./exp/diffsbdd-ddpo")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, args.log_level)
    logger.info(f"Loading config from: {args.config_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Save config to output directory
    config_save_path = os.path.join(output_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Config saved to: {config_save_path}")
    
    # Override config with command line arguments
    if args.save_path:
        config["output"] = config.get("output", {})
        config["output"]["dir"] = args.save_path
    
    if args.resume:
        config["resume"] = True
        config["checkpoint_path"] = args.checkpoint_path
    
    # Add wandb config for DDPOTrainer - enable logging via wandb_stub
    config["wandb"] = {
        "wandb_name": os.path.basename(output_dir),
        "wandb_project": "diffsbdd-ddpo",
    }
    # Set wandb=True to enable logging in DDPOTrainer
    # wandb_stub will handle the actual logging to local files
    
    # Set device
    device = config["model"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Get dataset info
    dataset_info = dataset_params["crossdock"]
    logger.info(f"Dataset info: {dataset_info['atom_decoder']}")
    
    # ==========================================================================
    # Initialize model
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("Initializing DiffSBDDModel...")
    logger.info("=" * 60)
    
    model = DiffSBDDModel(
        checkpoint_path=config["model"]["checkpoint"],
        pocket_pdb_path=config["pocket"]["pdb_path"],
        ref_ligand_path=config["pocket"].get("ref_ligand", "C:CIP"),
        device=device,
        dataset_name="crossdock"
    )
    logger.info(f"Model loaded: T={model.T}, atom_nf={model.atom_nf}")
    logger.info(f"Pocket info: {model.get_pocket_info()}")
    
    # ==========================================================================
    # Initialize dataloader
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("Initializing DataLoader...")
    logger.info("=" * 60)
    
    dataloader = DiffSBDDDataLoader(
        config=config,
        dataset_info=dataset_info,
        device=device,
        num_batches=config["dataloader"].get("epochs", 100)
    )
    logger.info(f"Batch size: {dataloader.batch_size}")
    logger.info(f"Num batches: {dataloader.num_batches}")
    
    # ==========================================================================
    # Initialize rollout
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("Initializing Rollout...")
    logger.info("=" * 60)
    
    rollout = DiffSBDDRollout(model, config)
    logger.info("Rollout initialized")

    
    # ==========================================================================
    # Initialize rewards
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("Initializing Rewards...")
    logger.info("=" * 60)
    
    # Create individual rewards
    docking_config = config["reward"]["docking"]
    # UniDock requires PDBQT format; fall back to pdb_path if pdbqt not specified
    receptor_for_docking = config["pocket"].get("pdbqt_path",
                           config["pocket"]["pdb_path"])
    docking_reward = DockingReward(
        backend=docking_config.get("backend", "autodock_gpu"),
        receptor_file=receptor_for_docking,
        normalization_factor=docking_config.get("normalization_factor", -10.0),
        center_x=docking_config.get("center_x"),
        center_y=docking_config.get("center_y"),
        center_z=docking_config.get("center_z"),
        size_x=docking_config.get("size_x", 20.0),
        size_y=docking_config.get("size_y", 20.0),
        size_z=docking_config.get("size_z", 20.0),
        exhaustiveness=docking_config.get("exhaustiveness", 8),
        autobox_ligand=docking_config.get("autobox_ligand"),
        autobox_add=docking_config.get("autobox_add", 4.0),
        chunk_size=docking_config.get("chunk_size"),
        dataset_info=dataset_info,
        # UniDock-specific
        gpu_id=docking_config.get("gpu_id", 0),
        search_mode=docking_config.get("search_mode", "fast"),
        # AutoDock-GPU-specific
        fld_file=docking_config.get("fld_file"),
        nrun=docking_config.get("nrun", 20),
        heuristics=docking_config.get("heuristics", True),
    )
    
    # Log docking mode
    if docking_reward.autobox_ligand:
        logger.info(f"DockingReward: receptor={config['pocket']['pdb_path']}, mode=autobox")
    elif docking_reward.center_x is not None:
        logger.info(f"DockingReward: receptor={config['pocket']['pdb_path']}, mode=full_docking, "
                   f"center=({docking_reward.center_x}, {docking_reward.center_y}, {docking_reward.center_z})")
    else:
        logger.info(f"DockingReward: receptor={config['pocket']['pdb_path']}, mode=score_only")
    
    druglikeness_reward = DruglikenessReward(
        qed_weight=0.5,
        sa_weight=0.5,
        dataset_info=dataset_info
    )
    logger.info("DruglikenessReward: qed_weight=0.5, sa_weight=0.5")
    
    # Create composite reward
    reward_weights = config["reward"].get("weights", {"docking": 0.6, "druglikeness": 0.4})
    
    # Map config weights to reward instances
    rewards_dict = {}
    weights_dict = {}
    
    if "docking" in reward_weights:
        rewards_dict["docking"] = docking_reward
        weights_dict["docking"] = reward_weights["docking"]
    
    # Combine qed and sa into druglikeness
    druglikeness_weight = reward_weights.get("qed", 0) + reward_weights.get("sa", 0)
    if druglikeness_weight > 0 or "druglikeness" in reward_weights:
        rewards_dict["druglikeness"] = druglikeness_reward
        weights_dict["druglikeness"] = reward_weights.get("druglikeness", druglikeness_weight)
    
    composite_reward = CompositeReward(
        rewards=rewards_dict,
        weights=weights_dict,
        normalize_weights=True
    )
    logger.info(f"CompositeReward weights: {weights_dict}")
    
    # Wrap with stability-aware reward for DDPOTrainer compatibility
    rewarder = StabilityAwareReward(composite_reward, dataset_info)
    logger.info("StabilityAwareReward wrapper applied")
    
    # ==========================================================================
    # Initialize filter
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("Initializing Filter...")
    logger.info("=" * 60)
    
    filters = DiffSBDDFilter(
        dataset_info=dataset_info,
        enable_filtering=True,
        enable_penalty=True
    )
    logger.info("DiffSBDDFilter initialized")
    
    # ==========================================================================
    # Initialize actor
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("Initializing Actor...")
    logger.info("=" * 60)
    
    actor = EDMActor(model, config)
    logger.info("EDMActor initialized")
    
    # ==========================================================================
    # Initialize Ray with resource configuration
    # ==========================================================================
    if not ray.is_initialized():
        ray_config = config.get("ray", {})
        num_cpus = ray_config.get("num_cpus", None)  # None = use all available
        num_gpus = ray_config.get("num_gpus", 0)     # 0 = no GPU for Ray tasks
        
        ray.init(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            ignore_reinit_error=True
        )
        logger.info(f"Ray initialized: num_cpus={num_cpus or 'auto'}, num_gpus={num_gpus}")
    
    # ==========================================================================
    # Initialize wandb logging (via wandb_stub)
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("Initializing Logging...")
    logger.info("=" * 60)
    
    # Initialize wandb before DDPOTrainer (it will detect existing run)
    wandb.init(
        project=config["wandb"]["wandb_project"],
        name=config["wandb"]["wandb_name"],
        config=config
    )
    logger.info(f"Logging to: {wandb.run.dir}")
    
    # Mark wandb as already initialized for DDPOTrainer
    # DDPOTrainer checks config.get("wandb", False) to enable logging
    # We set it to True so it uses our initialized wandb_stub
    
    # ==========================================================================
    # Initialize trainer
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("Initializing DDPOTrainer...")
    logger.info("=" * 60)
    
    trainer = DDPOTrainer(
        config=config,
        model=model,
        dataset_info=dataset_info,
        device=device,
        dataloader=dataloader,
        rollout=rollout,
        rewarder=rewarder,
        filters=filters,
        actor=actor
    )
    logger.info("DDPOTrainer initialized")
    
    # ==========================================================================
    # Start training
    # ==========================================================================
    logger.info("=" * 60)
    logger.info("Starting DDPO Training...")
    logger.info("=" * 60)
    
    try:
        trainer.fit()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    finally:
        trainer.clean_up()
        wandb.finish()
        logger.info("Training completed")


if __name__ == "__main__":
    main()
