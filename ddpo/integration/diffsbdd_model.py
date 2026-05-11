"""
DiffSBDDModel: Adapter wrapping DiffSBDD for RLPF interface.

This module provides a wrapper around DiffSBDD's LigandPocketDDPM model
that implements the RLPF BaseModel interface, enabling integration with
the DDPO training framework.

Key features:
- Loads DiffSBDD checkpoint and pocket data
- Implements RLPF-compatible sample(), sample_p_zs_given_zt(), compute_log_p_zs_given_zt()
- Handles format conversion between DiffSBDD and RLPF tensor formats
- Caches pocket data for efficient batch processing
"""

import logging
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from Bio.PDB import PDBParser

# Add DiffSBDD to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "DiffSBDD"))

from DiffSBDD.lightning_modules import LigandPocketDDPM
from DiffSBDD.constants import dataset_params, FLOAT_TYPE, INT_TYPE
from DiffSBDD import utils as diffsbdd_utils

# Import RLPF BaseModel
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "RLPF-verl_diffusion"))
from verl_diffusion.model.base import BaseModel

# Import mask conversion utilities
from ddpo.integration.mask_utils import (
    batch_mask_to_node_mask,
    node_mask_to_batch_mask,
    get_nodes_per_batch,
    create_batch_mask_from_counts,
)


class DiffSBDDModel(nn.Module, BaseModel):
    """
    Adapter wrapping DiffSBDD's LigandPocketDDPM for RLPF interface.
    
    This class implements the BaseModel interface required by RLPF's
    EDMActor and EDMRollout components, while internally using DiffSBDD's
    ConditionalDDPM for molecular generation.
    
    Attributes:
        model: LigandPocketDDPM instance (PyTorch Lightning module)
        ddpm: ConditionalDDPM instance (core diffusion model)
        residues: BioPython residue objects for the pocket
        pocket: Cached pocket data dict with x, one_hot, size, mask
        T: Number of diffusion timesteps
        n_dims: Number of spatial dimensions (3)
        atom_nf: Number of atom features (ligand atom types)
        aa_nf: Number of amino acid features (pocket residue types)
        dataset_info: Dataset configuration from DiffSBDD constants
    """

    def forward(self, mode: str, **kwargs):
        """Dispatch training forwards through the module call path.

        This explicit forward entry is required for DDP training semantics:
        PPO loss computation must enter through the wrapped module call
        (``DDP(module)(...)``) rather than bypassing DDP via direct helper
        method calls on the bare module.
        """
        if mode == "ddpo_log_prob":
            return self.forward_ddpo_log_prob(**kwargs)
        raise ValueError(f"Unsupported DiffSBDDModel forward mode: {mode}")
    
    def __init__(
        self,
        checkpoint_path: str,
        pocket_pdb_path: str,
        ref_ligand_path: str,
        device: str = "cuda",
        dataset_name: str = "crossdock",
    ):
        """
        Initialize DiffSBDDModel with checkpoint and pocket data.
        
        Args:
            checkpoint_path: Path to DiffSBDD checkpoint (.ckpt file)
            pocket_pdb_path: Path to PDB file containing the protein
            ref_ligand_path: Path to reference ligand (SDF file or chain:resi format)
                            Used to define the binding pocket
            device: Device to load model on ("cuda" or "cpu")
            dataset_name: Dataset name for constants ("crossdock" or "bindingmoad")
        """
        super().__init__()
        
        self.device = device
        self.dataset_name = dataset_name
        self.dataset_info = dataset_params[dataset_name]
        
        # Load DiffSBDD model from checkpoint
        self.model = self._load_model(checkpoint_path)
        self.model.to(device)
        # NOTE: Do NOT call model.eval() here!
        # In RL training, we need gradients for policy optimization.
        # The model should remain in training mode.
        self.model.train()  # Explicitly set to training mode
        
        # Store reference to ConditionalDDPM
        # NOTE: Use object.__setattr__ to avoid registering ddpm as a second
        # nn.Module submodule. self.model.ddpm is already registered under
        # self.model; registering it again as self.ddpm would cause Lightning
        # to serialize the same weights twice (244 tensors instead of 122).
        object.__setattr__(self, 'ddpm', self.model.ddpm)
        
        # Extract key attributes from model
        self.T = self.model.T  # Number of diffusion steps
        self.n_dims = self.model.x_dims  # Spatial dimensions (3)
        self.atom_nf = self.model.atom_nf  # Ligand atom types
        self.aa_nf = self.model.aa_nf  # Pocket residue types
        
        # Load pocket data
        self.residues = self._load_pocket_residues(pocket_pdb_path, ref_ligand_path)
        
        # Cache pocket data for single sample (will be replicated for batches)
        self._pocket_single = self.model.prepare_pocket(self.residues, repeats=1)
        
        # Store paths for reference
        self.checkpoint_path = checkpoint_path
        self.pocket_pdb_path = pocket_pdb_path
        self.ref_ligand_path = ref_ligand_path
    
    def _load_model(self, checkpoint_path: str) -> LigandPocketDDPM:
        """
        Load LigandPocketDDPM from checkpoint.
        
        Args:
            checkpoint_path: Path to .ckpt file
            
        Returns:
            Loaded LigandPocketDDPM model
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        model = LigandPocketDDPM.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device
        )
        return model
    
    def _load_pocket_residues(
        self, 
        pdb_path: str, 
        ref_ligand: str
    ) -> List:
        """
        Load pocket residues from PDB using reference ligand.
        
        Uses DiffSBDD's get_pocket_from_ligand() to extract residues
        within distance cutoff of the reference ligand.
        
        Args:
            pdb_path: Path to PDB file
            ref_ligand: Reference ligand (SDF path or chain:resi format)
            
        Returns:
            List of BioPython residue objects
        """
        if not Path(pdb_path).exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")
        
        # Parse PDB structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        pdb_model = structure[0]  # First model
        
        # Get pocket residues using DiffSBDD utility
        residues = diffsbdd_utils.get_pocket_from_ligand(
            pdb_model, 
            ref_ligand,
            dist_cutoff=8.0  # Default cutoff in DiffSBDD
        )
        
        if len(residues) == 0:
            raise ValueError(
                f"No pocket residues found within 8Å of ligand {ref_ligand}"
            )
        
        return residues
    
    def load(self, model_path: str) -> None:
        """
        Load model weights from path.
        
        Args:
            model_path: Path to checkpoint file
        """
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.ddpm = self.model.ddpm
    
    def to(self, device: str) -> "DiffSBDDModel":
        """
        Move model to specified device.
        
        Args:
            device: Target device ("cuda" or "cpu")
            
        Returns:
            Self for method chaining
        """
        self.device = device
        self.model.to(device)
        
        # Move cached pocket data
        self._pocket_single = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self._pocket_single.items()
        }
        
        return self
    
    def eval(self) -> "DiffSBDDModel":
        """Set model to evaluation mode."""
        self.model.eval()
        return self
    
    def train(self, mode: bool = True) -> "DiffSBDDModel":
        """Set model training mode."""
        self.model.train(mode)
        return self
    
    def get_pocket_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded pocket.
        
        Returns:
            Dict with pocket statistics
        """
        return {
            "num_residues": len(self.residues),
            "pocket_size": self._pocket_single["size"].item(),
            "pocket_x_shape": tuple(self._pocket_single["x"].shape),
            "pocket_one_hot_shape": tuple(self._pocket_single["one_hot"].shape),
        }

    def _log_cuda_memory(self, stage: str, device: torch.device | None = None) -> None:
        if not torch.cuda.is_available():
            return
        try:
            current_device = device if device is not None else torch.device("cuda", torch.cuda.current_device())
            allocated_mb = torch.cuda.memory_allocated(current_device) / (1024 ** 2)
            reserved_mb = torch.cuda.memory_reserved(current_device) / (1024 ** 2)
            max_allocated_mb = torch.cuda.max_memory_allocated(current_device) / (1024 ** 2)
            logging.getLogger("DiffSBDD-DDPO").info(
                "[Memory][model] %s | device=%s allocated=%.1fMB reserved=%.1fMB max_allocated=%.1fMB",
                stage,
                current_device,
                allocated_mb,
                reserved_mb,
                max_allocated_mb,
            )
        except Exception:
            logging.getLogger("DiffSBDD-DDPO").exception("Failed to log CUDA memory at stage %s", stage)
    
    # =========================================================================
    # Task 4.2: get_mask and format conversion methods
    # =========================================================================
    
    def get_mask(
        self, 
        nodesxsample: torch.Tensor, 
        batch_size: int, 
        max_n_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate RLPF-format node_mask and edge_mask.
        
        This method is called by EDMActor.calculate_loss() to create masks
        for the policy gradient computation. The implementation follows
        RLPF EDMModel.get_mask() exactly for compatibility.
        
        Args:
            nodesxsample: [B] tensor or list with number of nodes per sample
            batch_size: Number of samples in batch
            max_n_nodes: Maximum number of nodes (for padding)
            
        Returns:
            node_mask: [B, N, 1] tensor, 1=valid node, 0=padding
            edge_mask: [B*N*N, 1] tensor for edge masking (no self-loops)
            
        Note:
            DiffSBDD doesn't use edge_mask internally, but we generate it
            for interface compatibility with RLPF EDMActor.
        """
        # Create node_mask [B, N] - following RLPF EDMModel exactly
        node_mask = torch.zeros(batch_size, max_n_nodes)
        for i in range(batch_size):
            # Handle both tensor and list inputs
            n_nodes = nodesxsample[i]
            if isinstance(n_nodes, torch.Tensor):
                n_nodes = n_nodes.item()
            node_mask[i, 0:int(n_nodes)] = 1
        
        # Create edge_mask [B, N, N] -> [B*N*N, 1]
        # edge_mask[b, i, j] = 1 if both node i and j are valid and i != j
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)  # [B, N, N]
        
        # Remove self-loops (diagonal) - following RLPF EDMModel exactly
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        
        # Reshape to [B*N*N, 1]
        edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1)
        
        # Add last dimension to node_mask [B, N, 1]
        node_mask = node_mask.unsqueeze(2)
        
        return node_mask, edge_mask
    
    def _rlpf_to_diffsbdd_format(
        self, 
        x: torch.Tensor, 
        node_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert RLPF padded format to DiffSBDD flat format.
        
        RLPF uses [B, N, D] padded tensors where invalid positions are zeros.
        DiffSBDD uses [total_nodes, D] flat tensors with batch_mask indices.
        
        This conversion is necessary because:
        - RLPF EDMActor works with padded batches for efficient GPU computation
        - DiffSBDD's ConditionalDDPM expects flat tensors with scatter operations
        
        Args:
            x: [B, N, D] padded tensor from RLPF
            node_mask: [B, N, 1] mask tensor (1=valid, 0=padding)
            
        Returns:
            x_flat: [total_nodes, D] flat tensor for DiffSBDD
            batch_mask: [total_nodes] tensor with batch indices (long type)
            
        Example:
            >>> x = torch.randn(2, 5, 3)  # 2 samples, max 5 nodes, 3D coords
            >>> node_mask = torch.tensor([[[1],[1],[1],[0],[0]],
            ...                           [[1],[1],[0],[0],[0]]])  # 3 and 2 nodes
            >>> x_flat, batch_mask = model._rlpf_to_diffsbdd_format(x, node_mask)
            >>> x_flat.shape  # [5, 3] - total 5 valid nodes
            >>> batch_mask  # tensor([0, 0, 0, 1, 1])
        """
        # Convert node_mask to batch_mask using utility function
        batch_mask = node_mask_to_batch_mask(node_mask)
        
        # Extract valid nodes using boolean mask
        valid_mask = node_mask.squeeze(-1).bool()  # [B, N]
        x_flat = x[valid_mask]  # [total_nodes, D]
        
        # Ensure batch_mask is on the same device as x
        batch_mask = batch_mask.to(x.device)
        
        return x_flat, batch_mask
    
    def _diffsbdd_to_rlpf_format(
        self, 
        x: torch.Tensor, 
        batch_mask: torch.Tensor, 
        max_nodes: int
    ) -> torch.Tensor:
        """
        Convert DiffSBDD flat format to RLPF padded format.
        
        DiffSBDD uses [total_nodes, D] flat tensors with batch_mask indices.
        RLPF uses [B, N, D] padded tensors where invalid positions are zeros.
        
        Args:
            x: [total_nodes, D] flat tensor from DiffSBDD
            batch_mask: [total_nodes] tensor with batch indices
            max_nodes: Maximum number of nodes N for padding
            
        Returns:
            x_padded: [B, N, D] padded tensor for RLPF
        """
        batch_size = batch_mask.max().item() + 1
        D = x.shape[-1] if x.dim() > 1 else 1
        
        # Handle 1D input
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        # Create zero-padded output tensor
        x_padded = torch.zeros(
            batch_size, max_nodes, D,
            dtype=x.dtype, device=x.device
        )
        
        # Get number of nodes per batch
        nodes_per_batch = get_nodes_per_batch(batch_mask)
        
        # Fill valid positions
        idx = 0
        for b in range(batch_size):
            n_nodes = nodes_per_batch[b].item()
            x_padded[b, :n_nodes] = x[idx:idx + n_nodes]
            idx += n_nodes
        
        return x_padded
    
    def _replicate_pocket(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Replicate cached pocket data to specified batch size.
        
        Uses DiffSBDD's prepare_pocket with repeats parameter to efficiently
        create batch-sized pocket data.
        
        Args:
            batch_size: Target batch size
            
        Returns:
            pocket: Dict with replicated pocket data:
                - 'x': [N_residues * batch_size, 3] coordinates
                - 'one_hot': [N_residues * batch_size, 23] residue types
                - 'size': [batch_size] number of residues per sample
                - 'mask': [N_residues * batch_size] batch indices
        """
        # Use DiffSBDD's prepare_pocket with repeats parameter
        pocket = self.model.prepare_pocket(self.residues, repeats=batch_size)
        
        # Move to correct device
        pocket = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in pocket.items()
        }
        
        return pocket
    
    # =========================================================================
    # BaseModel interface methods (to be implemented in subsequent tasks)
    # =========================================================================
    
    def sample(self, *args, **kwargs):
        """
        Sample from the model (Task 4.7).
        
        This method will implement the full sampling loop, calling
        sample_p_zs_given_zt iteratively and recording trajectories.
        """
        raise NotImplementedError("sample() will be implemented in Task 4.7")
    
    def compute_log_p_zs_given_zt(
        self,
        zs: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        node_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute log probability of zs given zt for Gaussian transition.
        
        Implements the Gaussian log probability formula for the diffusion
        transition p(zs | zt) = N(zs; mu, sigma^2).
        
        This method is called by:
        - sample_p_zs_given_zt() to compute log_prob during sampling
        - EDMActor.calculate_loss() indirectly via sample_p_zs_given_zt()
        
        Args:
            zs: [B, N, D] sampled latent at time s (RLPF format)
            mu: [B, N, D] mean of the Gaussian distribution
            sigma: Scalar, [B, 1, 1], or [B, N, D] standard deviation
            node_mask: [B, N, 1] mask tensor (1=valid, 0=padding)
            
        Returns:
            log_prob: [B] log probability for each sample in batch
            
        Note:
            - Uses simplified formula without constant terms (only relative
              probabilities matter for PPO ratio computation)
            - Detaches zs to prevent gradients flowing through sampling path
            - Feature dimension D = n_dims (3) + atom_nf (ligand atom types)
        """
        # Numerical stability: ensure sigma > epsilon
        epsilon = 1e-6
        
        # Handle different sigma shapes - ensure it can broadcast with [B, N, D]
        if sigma.dim() == 0:
            # Scalar sigma
            sigma_safe = torch.max(sigma, torch.tensor(epsilon, device=sigma.device))
        elif sigma.dim() == 1:
            # [B] -> [B, 1, 1] for broadcasting
            sigma_safe = sigma.unsqueeze(-1).unsqueeze(-1)
            sigma_safe = torch.max(sigma_safe, torch.tensor(epsilon, device=sigma.device))
        elif sigma.dim() == 2:
            # [B, 1] or [B, N] -> [B, N, 1] or [B, 1, 1]
            if sigma.shape[1] == 1:
                sigma_safe = sigma.unsqueeze(-1)  # [B, 1, 1]
            else:
                sigma_safe = sigma.unsqueeze(-1)  # [B, N, 1]
            sigma_safe = torch.max(sigma_safe, torch.tensor(epsilon, device=sigma.device))
        else:
            # Already [B, N, D] or similar
            sigma_safe = torch.max(sigma, torch.tensor(epsilon, device=sigma.device))
        
        # Detach zs to prevent gradients through sampling path
        # (gradients should only flow through mu and sigma from the network)
        zs_detached = zs.detach()
        
        # Compute log probability: -0.5 * ((zs - mu)^2 / sigma^2)
        # Apply node_mask to ignore padding positions
        log_p = -0.5 * ((zs_detached - mu) ** 2) / (sigma_safe ** 2)
        
        if node_mask is not None:
            log_p = log_p * node_mask
        
        # Sum over all dimensions except batch, then normalize
        # Normalization factor: number of valid elements = sum(node_mask) * D
        # D = feature dimension (n_dims + atom_nf for DiffSBDD)
        D = zs.shape[-1]  # Feature dimension
        
        # Sum over nodes and features: [B, N, D] -> [B]
        log_p_sum = log_p.sum(dim=tuple(range(1, log_p.ndim)))
        
        if node_mask is not None:
            # Count valid elements per sample
            valid_count = node_mask.sum(dim=tuple(range(1, node_mask.ndim))) * D
        else:
            # All elements are valid
            valid_count = zs.shape[1] * D
        
        # Normalize by number of valid elements
        log_prob = log_p_sum / (valid_count + epsilon)
        
        # NaN/Inf 保护：log_prob 异常会导致 PPO ratio 爆炸，进而导致 backward 挂死
        log_prob = torch.clamp(log_prob, min=-100.0, max=0.0)
        log_prob = torch.where(torch.isfinite(log_prob), log_prob,
                               torch.full_like(log_prob, -100.0))
        
        return log_prob
    
    def sample_p_xh_given_z0(
        self,
        z0: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        context: torch.Tensor = None,
        fix_noise: bool = False,
        prev_sample: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample final x and h from z0 (last diffusion step).
        
        This method implements the final step of the diffusion process,
        converting the latent z0 to actual molecular coordinates (x) and
        atom types (h). It adapts DiffSBDD's sample_p_xh_given_z0 to
        the RLPF interface expected by EDMRollout.
        
        Args:
            z0: [B, N, D] latent at time 0 (RLPF format)
            node_mask: [B, N, 1] node mask (1=valid, 0=padding)
            edge_mask: [B*N*N, 1] edge mask (not used by DiffSBDD)
            context: Optional context tensor (not used)
            fix_noise: If True, use fixed noise for reproducibility
            prev_sample: [B, N, D] if provided, compute log_prob for this sample
            
        Returns:
            x: [B, N, 3] final atomic coordinates
            h: dict with 'integer' and 'categorical' keys for atom types
               h['categorical']: [B, N, num_atom_types] one-hot encoded
               h['integer']: [B, N, 1] integer charges (zeros for DiffSBDD)
            mu: [B, N, D] mean of the Gaussian distribution
            sigma: Standard deviation (scalar or broadcastable)
            log_prob: [B] log probability of the sample
            zeros: [B, 1] zero tensor (timestep indicator)
            xh: [B, N, D] combined x and h before splitting
            
        Note:
            - DiffSBDD returns (x_lig, h_lig, x_pocket, h_pocket) - 4 values
            - RLPF expects (x, h, mu, sigma, log_prob, zeros, xh) - 7 values
            - h is returned as dict for RLPF compatibility, but DiffSBDD
              doesn't use charges, so h['integer'] is zeros
        """
        import torch.nn.functional as F
        
        batch_size = z0.shape[0]
        max_nodes = z0.shape[1]
        device = z0.device
        
        # =====================================================================
        # Step 1: Convert RLPF format to DiffSBDD format
        # =====================================================================
        z0_flat, ligand_mask = self._rlpf_to_diffsbdd_format(z0, node_mask)
        
        # =====================================================================
        # Step 2: Prepare pocket data
        # =====================================================================
        pocket = self._replicate_pocket(batch_size)
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)
        pocket_mask = pocket['mask']
        
        # =====================================================================
        # Step 3: Compute gamma_0 and sigma_x
        # =====================================================================
        t_zeros = torch.zeros(size=(batch_size, 1), device=device)
        gamma_0 = self.ddpm.gamma(t_zeros)
        
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.ddpm.SNR(-0.5 * gamma_0)
        
        # =====================================================================
        # Step 4: Neural network forward pass
        # =====================================================================
        # Convert t_zeros to scalar (all values are 0)
        t_scalar = 0.0
        net_out_lig, _ = self.ddpm.dynamics(
            z0_flat, xh0_pocket, t_scalar, ligand_mask, pocket_mask
        )
        
        # =====================================================================
        # Step 5: Compute mu_x (mean for final sampling)
        # =====================================================================
        mu_x_flat = self.ddpm.compute_x_pred(net_out_lig, z0_flat, gamma_0, ligand_mask)
        
        # =====================================================================
        # Step 6: Sample xh with zero-centered coordinates
        # =====================================================================
        xh_flat, _ = self.ddpm.sample_normal_zero_com(
            mu_x_flat, xh0_pocket, sigma_x, ligand_mask, pocket_mask, fix_noise
        )
        
        # =====================================================================
        # Step 7: Unnormalize to get final x and h
        # =====================================================================
        x_lig_flat, h_lig_flat = self.ddpm.unnormalize(
            xh_flat[:, :self.n_dims],
            z0_flat[:, self.n_dims:]
        )
        
        # Convert h to one-hot
        h_lig_onehot_flat = F.one_hot(
            torch.argmax(h_lig_flat, dim=1), 
            self.atom_nf
        ).float()
        
        # =====================================================================
        # Step 8: Convert back to RLPF format
        # =====================================================================
        x = self._diffsbdd_to_rlpf_format(x_lig_flat, ligand_mask, max_nodes)
        h_cat = self._diffsbdd_to_rlpf_format(h_lig_onehot_flat, ligand_mask, max_nodes)
        mu = self._diffsbdd_to_rlpf_format(mu_x_flat, ligand_mask, max_nodes)
        xh = self._diffsbdd_to_rlpf_format(xh_flat, ligand_mask, max_nodes)
        
        # Apply node_mask to h_cat
        h_cat = h_cat * node_mask
        
        # Create h dict (RLPF format)
        # DiffSBDD doesn't use charges, so h['integer'] is zeros
        h_int = torch.zeros(batch_size, max_nodes, 1, device=device)
        h = {'integer': h_int, 'categorical': h_cat}
        
        # =====================================================================
        # Step 9: Compute log probability
        # =====================================================================
        if prev_sample is not None:
            log_prob = self.compute_log_p_zs_given_zt(
                prev_sample, mu, sigma_x, node_mask
            )
        else:
            log_prob = self.compute_log_p_zs_given_zt(
                xh, mu, sigma_x, node_mask
            )
        
        # =====================================================================
        # Step 10: Prepare zeros tensor (timestep indicator)
        # =====================================================================
        zeros = t_zeros  # [B, 1]
        
        # Squeeze sigma for return (RLPF expects scalar or [B])
        sigma_out = sigma_x.squeeze(-1) if sigma_x.dim() > 1 else sigma_x
        
        return x, h, mu, sigma_out, log_prob, zeros, xh
    
    def sample_p_zs_given_zt(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        zt: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        context: torch.Tensor = None,
        fix_noise: bool = False,
        prev_sample: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample zs given zt for single diffusion step.
        
        This method implements the reverse diffusion step p(zs | zt),
        adapting DiffSBDD's ConditionalDDPM.sample_p_zs_given_zt to
        the RLPF interface expected by EDMActor.
        
        Args:
            s: [B, 1] normalized time step s (target, s < t)
            t: [B, 1] normalized time step t (current)
            zt: [B, N, D] latent at time t (RLPF format)
            node_mask: [B, N, 1] node mask (1=valid, 0=padding)
            edge_mask: [B*N*N, 1] edge mask (not used by DiffSBDD)
            context: Optional context tensor (not used, for interface compatibility)
            fix_noise: If True, use fixed noise for reproducibility
            prev_sample: [B, N, D] if provided, compute log_prob for this sample
                        instead of the newly sampled zs (used by Actor)
            
        Returns:
            zs: [B, N, D] sampled latent at time s
            log_prob: [B] log probability of the sample
            mu: [B, N, D] mean of the Gaussian distribution
            sigma: Gaussian standard deviation (scalar or broadcastable)
            
        Note:
            - Internally converts between RLPF [B,N,D] and DiffSBDD [total_nodes,D] formats
            - Pocket data is automatically replicated to match batch size
            - Coordinates are kept zero-centered via DiffSBDD's sample_normal_zero_com
        """
        batch_size = zt.shape[0]
        max_nodes = zt.shape[1]
        mu_flat, sigma, ligand_mask, pocket_mask, xh0_pocket = self._compute_reverse_transition_stats(
            s=s,
            t=t,
            zt=zt,
            node_mask=node_mask,
        )

        # =====================================================================
        # Step 8: Sample zs (with zero-centered coordinates)
        # =====================================================================
        zs_flat, _ = self.ddpm.sample_normal_zero_com(
            mu_flat, xh0_pocket, sigma, ligand_mask, pocket_mask, fix_noise
        )
        
        # =====================================================================
        # Step 9: Convert back to RLPF format
        # =====================================================================
        zs = self._diffsbdd_to_rlpf_format(zs_flat, ligand_mask, max_nodes)
        mu = self._diffsbdd_to_rlpf_format(mu_flat, ligand_mask, max_nodes)
        
        # =====================================================================
        # Step 10: Compute log probability
        # =====================================================================
        # If prev_sample is provided (Actor training), compute log_prob for it
        # Otherwise, compute log_prob for the newly sampled zs
        if prev_sample is not None:
            log_prob = self.compute_log_p_zs_given_zt(
                prev_sample, mu, sigma, node_mask
            )
        else:
            log_prob = self.compute_log_p_zs_given_zt(
                zs, mu, sigma, node_mask
            )
        
        return zs, log_prob, mu, sigma
    
    def _compute_reverse_transition_stats(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        zt: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute reverse-step Gaussian parameters for p(z_s | z_t).

        This helper preserves the existing diffusion semantics while exposing
        the trainable network forward through ``forward()`` so DDP can manage
        gradient synchronization for PPO updates.
        """
        batch_size = zt.shape[0]

        # RLPF: [B, N, D] padded -> DiffSBDD: [total_nodes, D] flat
        zt_flat, ligand_mask = self._rlpf_to_diffsbdd_format(zt, node_mask)

        pocket = self._replicate_pocket(batch_size)
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)
        pocket_mask = pocket['mask']

        gamma_s = self.ddpm.gamma(s)
        gamma_t = self.ddpm.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.ddpm.sigma_and_alpha_t_given_s(
            gamma_t,
            gamma_s,
            zt_flat,
        )

        sigma_s = self.ddpm.sigma(gamma_s, target_tensor=zt_flat)
        sigma_t = self.ddpm.sigma(gamma_t, target_tensor=zt_flat)

        if t.numel() == 1:
            t_scalar = t.item()
        elif torch.all(t == t[0]):
            t_scalar = t[0].item()
        else:
            t_scalar = t[0].item()

        eps_t_lig, _ = self.ddpm.dynamics(
            zt_flat,
            xh0_pocket,
            t_scalar,
            ligand_mask,
            pocket_mask,
        )

        mu_flat = zt_flat / alpha_t_given_s[ligand_mask] - (
            sigma2_t_given_s / alpha_t_given_s / sigma_t
        )[ligand_mask] * eps_t_lig
        sigma = sigma_t_given_s * sigma_s / sigma_t
        return mu_flat, sigma, ligand_mask, pocket_mask, xh0_pocket

    def forward_ddpo_log_prob(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        zt: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        prev_sample: torch.Tensor,
        context: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (log_prob, mu, sigma) under the current policy via forward.

        Returns a 3-tuple so that callers (PPO loss *and* optional KL penalty)
        can reuse the same reverse-transition computation without redundant
        network forward passes.

        ``edge_mask`` and ``context`` are accepted for interface stability but
        are not consumed by DiffSBDD's current reverse transition computation.
        """
        del edge_mask, context
        max_nodes = zt.shape[1]
        mu_flat, sigma, ligand_mask, _pocket_mask, _xh0_pocket = self._compute_reverse_transition_stats(
            s=s,
            t=t,
            zt=zt,
            node_mask=node_mask,
        )
        mu = self._diffsbdd_to_rlpf_format(mu_flat, ligand_mask, max_nodes)
        log_prob = self.compute_log_p_zs_given_zt(prev_sample, mu, sigma, node_mask)
        return log_prob, mu, sigma

    # =========================================================================
    # Task 4.7: sample method (full sampling loop)
    # =========================================================================
    
    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        n_nodes: int,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        context: torch.Tensor = None,
        fix_noise: bool = False,
        timestep: int = 1000
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[int]]:
        """
        Sample from the diffusion model (full sampling loop).
        
        This method implements the complete reverse diffusion process,
        iteratively sampling p(z_s | z_t) for t = T, ..., 1, then
        sampling p(x, h | z_0) to get the final molecular structure.
        
        The implementation follows RLPF EDMModel.sample() interface while
        internally using DiffSBDD's ConditionalDDPM for the actual sampling.
        
        Args:
            n_samples: Number of molecules to generate
            n_nodes: Number of atoms per molecule (can be int or tensor)
            node_mask: [B, N, 1] node mask (1=valid, 0=padding)
            edge_mask: [B*N*N, 1] edge mask (not used by DiffSBDD)
            context: Optional context tensor (not used)
            fix_noise: If True, use fixed noise for reproducibility
            timestep: Number of diffusion timesteps (default 1000)
            
        Returns:
            x: [B, N, 3] final atomic coordinates
            h: dict with 'integer' and 'categorical' keys for atom types
               h['categorical']: [B, N, num_atom_types] one-hot encoded
               h['integer']: [B, N, 1] integer charges (zeros for DiffSBDD)
            latents: List of [B, N, D] latent tensors at each timestep
            logps: List of [B] log probabilities at each timestep
            timesteps: List of timestep indices

        Note:
            - Trajectory recording is essential for DDPO training
            - Coordinates are kept zero-centered via DiffSBDD's remove_mean_batch
            - h is returned as dict for RLPF compatibility
        """
        from torch_scatter import scatter_mean, scatter_add
        
        # Store original T and set new timestep
        original_T = self.T
        self.T = timestep
        
        try:
            device = node_mask.device
            max_nodes = node_mask.shape[1]
            self._log_cuda_memory("sample:start", device=device)
            
            # =====================================================================
            # Step 1: Initialize - Sample initial noise z_T in pocket center
            # =====================================================================
            # Replicate pocket data for batch
            pocket = self._replicate_pocket(n_samples)
            xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)
            pocket_mask = pocket['mask']
            
            # Create ligand mask in DiffSBDD format
            # Convert node_mask [B, N, 1] to batch_mask [total_nodes]
            ligand_mask = node_mask_to_batch_mask(node_mask).to(device)
            
            # Sample initial noise centered at pocket
            # mu_lig_x: center of pocket for each sample
            mu_lig_x = scatter_mean(pocket['x'], pocket_mask, dim=0)  # [B, 3]
            mu_lig_h = torch.zeros((n_samples, self.atom_nf), device=device)
            mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)  # [B, n_dims + atom_nf]
            
            # Expand mu_lig to flat format [total_nodes, D]
            mu_lig_flat = mu_lig[ligand_mask]
            
            # Initial sigma = 1
            sigma_init = torch.ones_like(pocket['size']).unsqueeze(1)
            
            # Sample z_T with zero center of mass
            z_flat, _ = self.ddpm.sample_normal_zero_com(
                mu_lig_flat, xh0_pocket, sigma_init, ligand_mask, pocket_mask, fix_noise
            )
            
            # Convert to RLPF format [B, N, D]
            z = self._diffsbdd_to_rlpf_format(z_flat, ligand_mask, max_nodes)
            
            # =====================================================================
            # Step 2: Initialize trajectory recording
            # =====================================================================
            latents = [z.clone()]
            logps = []
            timesteps_list = []
            
            # Progress tracking - 单行覆盖，仅 rank-0 打印
            _local_rank = int(__import__('os').environ.get('LOCAL_RANK', '0'))
            _is_rank0 = (_local_rank == 0)
            _epoch_info = getattr(self, '_progress_epoch', (0, 0))
            _epoch_str = f"Epoch {_epoch_info[0]}/{_epoch_info[1]}" if _epoch_info[1] > 0 else "Epoch ?"
            if _is_rank0:
                print(f"\r[{_epoch_str}] 生成中: {n_samples}分子/卡, 0% (0/{self.T}步)  ", end="", flush=True)
            progress_interval = max(1, self.T // 10)  # Print every 10%
            memory_interval = max(1, self.T // 5)
            
            # =====================================================================
            # Step 3: Iteratively sample p(z_s | z_t) for t = T, ..., 1
            # =====================================================================
            for s in reversed(range(0, self.T)):
                # Progress output - 仅 rank-0 单行覆盖
                if _is_rank0 and (s % progress_interval == 0 or s == self.T - 1):
                    progress = (self.T - s) / self.T * 100
                    print(f"\r[{_epoch_str}] 生成中: {progress:.0f}% ({self.T-s}/{self.T}步)  ", end="", flush=True)
                if s % memory_interval == 0 or s == self.T - 1:
                    self._log_cuda_memory(f"sample:step_{self.T - s}_of_{self.T}", device=device)
                s_array = torch.full((n_samples, 1), fill_value=s, device=device)
                t_array = s_array + 1
                s_array = s_array / self.T
                t_array = t_array / self.T
                
                # Sample z_s given z_t
                z, logp, mu, sigma = self.sample_p_zs_given_zt(
                    s_array, t_array, z, node_mask, edge_mask, 
                    context=context, fix_noise=fix_noise
                )
                
                # Record trajectory
                latents.append(z.clone())
                logps.append(logp)
                timesteps_list.append(s)
            
            if _is_rank0:
                print(f"\r[{_epoch_str}] 生成完成: {n_samples}分子/卡, {self.T}步 (100%)  ", flush=True)
            
            # =====================================================================
            # Step 4: Final step - sample p(x, h | z_0)
            # =====================================================================
            x, h, mu_final, sigma_final, logp_final, zeros, xh = self.sample_p_xh_given_z0(
                z, node_mask, edge_mask, context=context, fix_noise=fix_noise
            )
            self._log_cuda_memory("sample:after_final_decode", device=device)
            
            # Record final step
            latents.append(xh.clone())
            logps.append(logp_final)
            timesteps_list.append(0)
            
            # =====================================================================
            # Step 5: Verify zero center of mass and correct if needed
            # =====================================================================
            # Check CoM drift
            x_sum = (x * node_mask).sum(dim=1)  # [B, 3]
            n_valid = node_mask.sum(dim=1)  # [B, 1]
            cog = x_sum / (n_valid + 1e-8)  # [B, 3]
            max_cog = cog.abs().max().item()
            
            if max_cog > 5e-2:
                logging.getLogger("DiffSBDD-DDPO").debug(
                    f"CoG drift detected: {max_cog:.3f}, projecting positions"
                )
                # Remove mean to center coordinates
                x = x - cog.unsqueeze(1) * node_mask
        
        finally:
            # Always restore original T, even if an exception occurred
            self.T = original_T
        
        return x, h, latents, logps, timesteps_list
    
    def _assert_mean_zero_with_mask(
        self, 
        x: torch.Tensor, 
        node_mask: torch.Tensor,
        eps: float = 1e-4
    ) -> None:
        """
        Assert that the center of mass is approximately zero.
        
        Args:
            x: [B, N, 3] coordinate tensor
            node_mask: [B, N, 1] mask tensor
            eps: Tolerance for assertion
        """
        # Compute center of mass
        x_masked = x * node_mask
        x_sum = x_masked.sum(dim=1)  # [B, 3]
        n_valid = node_mask.sum(dim=1)  # [B, 1]
        cog = x_sum / (n_valid + 1e-8)
        
        max_cog = cog.abs().max().item()
        if max_cog > eps:
            logging.getLogger("DiffSBDD-DDPO").debug(
                f"Mean is not zero, max CoG = {max_cog:.6f}"
            )
    
    def _remove_mean_with_mask(
        self,
        x: torch.Tensor,
        node_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Remove center of mass from coordinates.
        
        Args:
            x: [B, N, 3] coordinate tensor
            node_mask: [B, N, 1] mask tensor
            
        Returns:
            x_centered: [B, N, 3] centered coordinates
        """
        x_masked = x * node_mask
        x_sum = x_masked.sum(dim=1, keepdim=True)  # [B, 1, 3]
        n_valid = node_mask.sum(dim=1, keepdim=True)  # [B, 1, 1]
        mean = x_sum / (n_valid + 1e-8)
        
        return x - mean * node_mask

    def parameters(self):
        """
        Return model parameters for optimizer.
        
        Required by RLPF's EDMActor for PPO training.
        
        Returns:
            Iterator over model parameters
        """
        return self.model.parameters()
    
    def to(self, device):
        """
        Move model to specified device.
        
        Args:
            device: Target device (cuda/cpu)
            
        Returns:
            self
        """
        self.device = device
        self.model.to(device)
        self.ddpm.to(device)
        
        # Move cached pocket data
        if hasattr(self, '_pocket_single'):
            self._pocket_single = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in self._pocket_single.items()
            }
        
        return self
