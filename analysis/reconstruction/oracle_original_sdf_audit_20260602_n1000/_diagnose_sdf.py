"""Diagnose SDF issues by trying to read each entry with RDKit."""
import os, sys
from rdkit import Chem

sdf_dir = os.path.dirname(os.path.abspath(__file__))
sdf_path = os.path.join(sdf_dir, 'original_vs_repaired_oracle_pairs.sdf')

suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
total = 0
ok = 0
problems = {}
first_problem = True

for i, mol in enumerate(suppl):
    total += 1
    if mol is None:
        problems[i] = "None mol (parse failure)"
    else:
        ok += 1
        # Check for sanitization
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            problems[i] = f"sanitize failure: {e}"
        else:
            continue

print(f"Total: {total}, OK: {ok}, Problems: {len(problems)}")
if problems:
    print(f"\nProblem entries (first 20):")
    for idx in sorted(problems.keys())[:20]:
        print(f"  Entry {idx}: {problems[idx]}")
        if first_problem and idx < len(suppl._fileNames if hasattr(suppl, '_fileNames') else []):
            # Try to show raw text around the entry
            pass
    print(f"\n... and {len(problems)-min(20, len(problems))} more")

# Also check if there are entries with invalid atom types or coordinates
print(f"\nCheck patterns in SDF...")
with open(sdf_path, 'r') as f:
    content = f.read()

# Check for common SDF issues
import re

# Check if any V2000 line has non-standard counts
v2000_lines = re.findall(r'^\s*\d+\s+\d+\s+.*V2000', content, re.MULTILINE)
print(f"Total V2000 entries: {len(v2000_lines)}")

# Check for empty title lines
empty_titles = content.count('\n\n') 
print(f"Empty title (double newline): about {empty_titles} occurrences")

# Check bond block between atom block and M END
# Look for entries where bond atom index exceeds num_atoms
entries = content.split('$$$$\n')
bond_range_errors = 0
for i, entry in enumerate(entries):
    if not entry.strip():
        continue
    # Find V2000 line
    m = re.search(r'^\s*(\d+)\s+(\d+)\s+.*V2000', entry, re.MULTILINE)
    if not m:
        continue
    na = int(m.group(1))
    nb = int(m.group(2))
    
    # Find bond block (after count line + na atom lines)
    lines = entry.split('\n')
    count_idx = None
    for j, l in enumerate(lines):
        if 'V2000' in l:
            count_idx = j
            break
    if count_idx is None:
        continue
    bond_start = count_idx + 1 + na
    bond_end = bond_start + nb
    
    for k in range(bond_start, min(bond_end, len(lines))):
        bl = lines[k].rstrip()
        if len(bl) < 6:
            continue
        try:
            a1 = int(bl[0:3])
            a2 = int(bl[3:6])
            if a1 < 1 or a1 > na or a2 < 1 or a2 > na:
                bond_range_errors += 1
                if bond_range_errors <= 5:
                    print(f"  Entry {i}: bond atom out of range: a1={a1}, a2={a2}, max={na}")
        except ValueError:
            pass

print(f"Bond atom range errors: {bond_range_errors}")

# Check atom symbol field
# In V2000, atom symbol is at position 31-33 (0-indexed: 31, 32, 33)
# Actually let's look for unusual elements
unusual = re.findall(r'^.{31}(..)', content, re.MULTILINE)
from collections import Counter
counter = Counter(u.strip() for u in unusual if u.strip())
print(f"\nAtom symbols found: {dict(counter.most_common(20))}")
