"""Read specific entries with RDKit and report exact errors."""
import os
from rdkit import Chem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

sdf_path = os.path.join(os.path.dirname(__file__), 'original_vs_repaired_oracle_pairs.sdf')

# Read all and identify failing entries
suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
fail_indices = []
for i, mol in enumerate(suppl):
    if mol is None:
        fail_indices.append(i)

print(f"Parse failures: {len(fail_indices)} indices")
print(f"First 15: {fail_indices[:15]}\n")

# Now read the failing entries individually to see exact SDF text
with open(sdf_path, 'r') as f:
    content = f.read()

entries = content.split('$$$$\n')

# Check the first few failing entries with verbose RDKit
from rdkit.Chem import rdmolfiles
from io import StringIO

for idx in fail_indices[:5]:
    e = entries[idx].strip()
    
    # Show raw header lines
    lines = e.split('\n')
    print(f'=== Entry {idx}: title=[{lines[0].strip()}]')
    print(f'  Total lines: {len(lines)}')
    
    # Try to parse
    block = e + '\n$$$$\n'
    
    # Check for \r issues
    has_r = '\r\n' in block or '\r' in block
    print(f'  Has \\r: {has_r}')
    
    # First 100 chars as hex
    first100 = e[:100]
    hex_chars = ' '.join(f'{ord(c):02x}' for c in first100[30:60])
    print(f'  Hex[30:60]: {hex_chars}')
    
    # Try parsing with error
    mol = Chem.MolFromMolBlock(block, removeHs=False, sanitize=False)
    if mol is None:
        print(f'  RDKit: MolFromMolBlock returned None')
    else:
        n = mol.GetNumAtoms()
        print(f'  RDKit: got {n} atoms')
    print()
