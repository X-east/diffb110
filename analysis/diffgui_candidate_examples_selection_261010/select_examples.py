"""Select traceable display examples from frozen candidates without new filters."""
from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write directly inspectable evidence tables."""
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ranking(row: dict) -> tuple:
    """Use docking then QED, raw SA and identifier as deterministic tie breaks."""
    return float(row['score']), -float(row['qed']), float(row['sa']), row['record_id']


def main() -> None:
    """Apply a predeclared illustration rule, preserving all candidate counts."""
    source = ROOT / 'analysis/diffgui_literature_screen_dock10_261010/epoch80_with_r1r5_candidates.csv'
    method = dict(purpose='Display examples, not a validated hit ranking or final shortlist',
                  input='All 119 frozen candidates; no additional chemical alert filtering',
                  dedup='Canonical isomeric SMILES; keep best docking record with deterministic tie breaks',
                  scaffold='RDKit Bemis-Murcko scaffold, includeChirality=False; retain atom and bond types',
                  common='Take 3 largest scaffold groups; ties: group best docking, then scaffold SMILES. Within each group, take best docking.',
                  metric='Sequentially take best docking, highest QED, lowest raw SA from remaining unshown scaffolds; keep all 6 scaffolds distinct.',
                  tie_break='For ties: docking ascending, QED descending, raw SA ascending, record identifier ascending',
                  exclusions='None beyond frozen physicochemical, docking <= -10 and R1-R5 criteria',
                  basis='RDKit https://www.rdkit.org/docs/GettingStartedInPython.html#murcko-decomposition')
    (HERE / 'selection_method.json').write_text(json.dumps(method, ensure_ascii=False, indent=2))
    with source.open(encoding='utf-8-sig', newline='') as handle:
        original = list(csv.DictReader(handle))
    unique = {}
    for row in sorted(original, key=ranking):
        mol = Chem.MolFromSmiles(row['smiles'])
        key = Chem.MolToSmiles(mol, isomericSmiles=True)
        if key not in unique:
            row = dict(row)
            row['canonical_isomeric_smiles'] = key
            row['display_scaffold'] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            unique[key] = row
    groups = defaultdict(list)
    for row in unique.values():
        groups[row['display_scaffold']].append(row)
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), float(min(item[1], key=ranking)['score']), item[0]))
    selected = []
    used = set()
    for index, (scaffold, members) in enumerate(ordered[:3], 1):
        row = dict(min(members, key=ranking))
        row['display_reason'] = f'高频骨架{index}（{len(members)}个）'
        row['role'] = f'common_scaffold_{index}'
        selected.append(row)
        used.add(scaffold)
    for role, title, key in [('best_docking', '其余骨架：对接最好', ranking),
                              ('highest_qed', '其余骨架：QED最高', lambda r: (-float(r['qed']), ranking(r))),
                              ('lowest_sa', '其余骨架：SA最低', lambda r: (float(r['sa']), ranking(r)))]:
        row = dict(min([r for r in unique.values() if r['display_scaffold'] not in used], key=key))
        row['display_reason'] = title
        row['role'] = role
        selected.append(row)
        used.add(row['display_scaffold'])
    for row in selected:
        row['scaffold_frequency'] = len(groups[row['display_scaffold']])
    assert len(used) == len(selected) == 6
    write_csv(HERE / 'selected_examples.csv', selected)
    write_csv(HERE / 'deduplicated_candidates.csv', list(unique.values()))
    write_csv(HERE / 'scaffold_groups.csv', [dict(scaffold=sc, count=len(members),
              best_docking_id=min(members, key=ranking)['record_id'], selected=sc in used)
              for sc, members in ordered])
    assets = HERE / 'assets'
    assets.mkdir(exist_ok=True)
    review = ['g6_224', 'g1_183', 'g6_175', 'g1_243']
    lookup = {r['record_id']: r for r in original}
    arial_file = subprocess.check_output(['fc-match', '-f', '%{file}', 'Arial'], text=True).strip()
    for row in selected + [lookup[k] for k in review]:
        mol = Chem.MolFromSmiles(row['smiles'])
        AllChem.Compute2DCoords(mol)
        drawing = rdMolDraw2D.MolDraw2DCairo(1000, 500)
        drawing.drawOptions().padding = .08
        drawing.drawOptions().baseFontSize = .8
        drawing.drawOptions().bondLineWidth = 3.5
        drawing.drawOptions().fontFile = arial_file
        drawing.DrawMolecule(mol)
        drawing.FinishDrawing()
        (assets / f"{row['record_id']}.png").write_bytes(drawing.GetDrawingText())
    canvas = Image.new('RGB', (1500, 800), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(arial_file, 23)
    for i, row in enumerate(selected):
        x, y = (i % 3)*500, (i // 3)*400
        img = Image.open(assets / f"{row['record_id']}.png").resize((500, 250))
        canvas.paste(img, (x, y+55))
        draw.text((x+15,y+10),f"{row['record_id']} | {row['role']}",font=font,fill='#064C76')
        draw.text((x+15,y+310),f"Dock {float(row['score']):.2f} | QED {float(row['qed']):.2f} | SA {float(row['sa']):.2f}",font=font,fill='#222222')
    canvas.save(HERE / 'selected_examples.png')
    facts = dict(input_count=len(original), unique_count=len(unique), scaffold_count=len(groups),
                 singleton_scaffolds=sum(len(v)==1 for v in groups.values()),
                 selected_ids=[r['record_id'] for r in selected],
                 shown_scaffold_members=sum(len(groups[sc]) for sc in used))
    (HERE / 'selection_summary.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2))
    data = dict(examples=facts['selected_ids'], review=review,
                molecules={r['record_id']:r for r in selected + [lookup[k] for k in review]}, selection=facts)
    (HERE / 'slide_data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
