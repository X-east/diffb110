"""Export individual molecular structures and source facts for review slides."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    """Draw fixed examples without altering their stored chemical identities."""
    source = ROOT / 'analysis/diffgui_literature_screen_dock10_261010'
    with (source / 'epoch80_with_r1r5_candidates.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = {r['record_id']: r for r in csv.DictReader(handle)}
    examples = ['g1_137', 'g2_119', 'g1_131', 'g6_186', 'g6_204', 'g4_226']
    review = ['g6_224', 'g1_183', 'g6_175', 'g1_243']
    assets = HERE / 'assets'
    assets.mkdir(exist_ok=True)
    selected = {}
    arial_file = subprocess.check_output(['fc-match', '-f', '%{file}', 'Arial'], text=True).strip()
    for key in examples + review:
        row = rows[key]
        molecule = Chem.MolFromSmiles(row['smiles'])
        assert molecule is not None
        AllChem.Compute2DCoords(molecule)
        drawing = rdMolDraw2D.MolDraw2DCairo(1000, 500)
        drawing.drawOptions().padding = 0.08
        drawing.drawOptions().baseFontSize = 0.8
        drawing.drawOptions().bondLineWidth = 3.5
        drawing.drawOptions().fontFile = arial_file
        drawing.DrawMolecule(molecule)
        drawing.FinishDrawing()
        (assets / f'{key}.png').write_bytes(drawing.GetDrawingText())
        selected[key] = {field: row[field] for field in ['record_id', 'smiles', 'score', 'mw', 'qed', 'sa', 'logp']}
    summary = json.loads((ROOT / 'analysis/diffgui_final_candidate_structure_review_261010/summary.json').read_text())
    (HERE / 'slide_data.json').write_text(json.dumps(dict(examples=examples, review=review, molecules=selected,
                                                       summary=summary), ensure_ascii=False, indent=2))
    print('Prepared 10 molecular structure images and source facts.')


if __name__ == '__main__':
    main()
