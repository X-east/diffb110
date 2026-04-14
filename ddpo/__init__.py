"""
DDPO Integration Layer for DiffSBDD-RLPF.

This package provides the integration between DiffSBDD (E(n)-equivariant diffusion model)
and RLPF (verl_diffusion) framework for DDPO training.
"""

__version__ = "0.1.0"

# ============================================================================
# BioPython compatibility fix for newer versions
# In BioPython >= 1.80, three_to_one was renamed to protein_letters_3to1
# ============================================================================
import sys
try:
    from Bio.PDB.Polypeptide import three_to_one
except ImportError:
    # BioPython >= 1.80: use new name and create alias
    from Bio.PDB.Polypeptide import protein_letters_3to1 as three_to_one
    from Bio.PDB import Polypeptide
    # Patch the module so DiffSBDD imports work
    Polypeptide.three_to_one = three_to_one

# Install wandb stub before any DiffSBDD imports to avoid wandb dependency
from ddpo.utils import wandb_stub
if 'wandb' not in sys.modules:
    sys.modules['wandb'] = wandb_stub
