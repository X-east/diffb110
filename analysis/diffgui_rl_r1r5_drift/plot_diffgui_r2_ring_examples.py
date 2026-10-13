"""DiffGui R2 ring-system violation example figure (P1 evidence for thesis Ch.3).

Picks DiffGui 500-step benchmark molecules that (a) pass the standard validity
gate yet (b) violate R2 ring-system rule while having >=2 aromatic rings. This
visualizes the thesis inversion point: a high aromatic-ring rate can *mask*
ring-system deformity, so hard metrics (aromatic ring count, validity) look
healthy while R1-R5 pathology is severe.

Source SDF: 5-repeat 500-step benchmark, rep4 aggregated diffgui.sdf (867 mols).
Output: work_records/images/diffgui_r2_ring_example.png
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem
from rdkit.Chem import Draw

from scripts.offline_rl.structure_pathology_rules import (
    compute_structure_pathology_from_smiles,
)

SDF = PROJECT_ROOT / "analysis/benchmark/benchmark5/five_model_repeats_500steps_260826/rep4_seed22_29/aggregated/diffgui.sdf"
OUT = PROJECT_ROOT / "work_records/images/diffgui_r2_ring_example.png"
N_EXAMPLES = 6


def count_aromatic_rings(mol: Chem.Mol) -> int:
    ri = mol.GetRingInfo()
    return sum(
        1
        for ring in ri.AtomRings()
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
    )


def main() -> None:
    suppl = Chem.SDMolSupplier(str(SDF), removeHs=False)
    examples: list[tuple[Chem.Mol, int]] = []
    total_r2 = 0
    for mol in suppl:
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:  # noqa: BLE001
            continue
        smi = Chem.MolToSmiles(mol)
        facts = compute_structure_pathology_from_smiles(smi)
        if facts is None:
            continue
        if facts.r2_ring_system_violation:
            total_r2 += 1
        arom = count_aromatic_rings(mol)
        if facts.r2_ring_system_violation and arom >= 2:
            examples.append((mol, arom))

    examples.sort(key=lambda x: -x[1])
    selected = examples[:N_EXAMPLES]
    print(f"R2 ring-system violators in SDF: {total_r2}")
    print(f"R2 violators with >=2 aromatic rings: {len(examples)}; showing {len(selected)}")
    for m, a in selected:
        print(f"  arom_rings={a}  {Chem.MolToSmiles(m)}")

    mols = [m for m, _ in selected]
    labels = [f"arom rings={a}\nR2 ring-system viol" for _, a in selected]
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=3,
        legends=labels,
        subImgSize=(320, 300),
        useSVG=False,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(OUT))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
